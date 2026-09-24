"""All persistent and interchange data models, plus JSON Schema export and drift checking.

Every model forbids unknown fields. Models that other workflows consume (plans, traces, skills,
manifests, checkpoints, events, tool descriptors) are exported to ``/schemas`` and a drift check
fails the build when the committed files differ from the generated ones.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from crystallizer.config import Config

Scalar = str | int | float | bool
ArgValue = str | int | bool | list[str]

IDENT_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
KIND_PATTERN = r"^[A-Za-z][A-Za-z0-9_.\-]{0,63}$"
TASK_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$"
FIELD_PATTERN = r"^(task_kind|step_kind|step_index|params\.[A-Za-z_][A-Za-z0-9_]{0,63})$"
TIER_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"


class Strict(BaseModel):
    """Base model: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


class Frozen(BaseModel):
    """Base model for immutable values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- actions & steps


class Action(Frozen):
    """One tool invocation: a tool name and its arguments."""

    tool: str = Field(min_length=1, max_length=64)
    args: dict[str, ArgValue] = Field(default_factory=dict)


class Situation(Frozen):
    """What is known before a step is routed. Field paths are dotted (``params.name``)."""

    task_kind: str
    step_kind: str
    step_index: int = Field(default=0, ge=0)
    params: dict[str, Scalar] = Field(default_factory=dict)

    def lookup(self, field: str) -> tuple[bool, Scalar | None]:
        """Return ``(exists, value)`` for a dotted field path."""
        if field == "task_kind":
            return True, self.task_kind
        if field == "step_kind":
            return True, self.step_kind
        if field == "step_index":
            return True, self.step_index
        if field.startswith("params."):
            name = field[len("params.") :]
            if name in self.params:
                return True, self.params[name]
        return False, None

    def fields(self) -> dict[str, Scalar]:
        """Return every field as a flat ``{dotted_path: value}`` mapping."""
        flat: dict[str, Scalar] = {
            "task_kind": self.task_kind,
            "step_kind": self.step_kind,
            "step_index": self.step_index,
        }
        for name, value in self.params.items():
            flat[f"params.{name}"] = value
        return flat


# --------------------------------------------------------------------------- plans


class TaskState(StrEnum):
    """Lifecycle state of a task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class StepSpec(Strict):
    """A planned step: its kind (used for routing and mining) and a short description."""

    kind: str = Field(pattern=KIND_PATTERN)
    description: str = Field(default="", max_length=2000)


class Task(Strict):
    """A unit of work in the plan DAG."""

    id: str = Field(pattern=TASK_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    kind: str = Field(pattern=KIND_PATTERN)
    params: dict[str, Scalar] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    acceptance_commands: list[list[str]] = Field(default_factory=list)
    steps: list[StepSpec] = Field(default_factory=list)
    state: TaskState = TaskState.PENDING
    attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Task:
        for name in self.params:
            if not _matches(IDENT_PATTERN, name):
                raise ValueError(f"param name {name!r} must be an identifier")
        for command in self.acceptance_commands:
            if not command or not all(command):
                raise ValueError("acceptance commands must be non-empty argv lists")
        return self


class Plan(Strict):
    """A goal decomposed into a DAG of tasks."""

    schema_version: int = 1
    goal: str = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)

    def task(self, task_id: str) -> Task:
        """Return the task with ``task_id`` (raises ``KeyError`` if absent)."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)


# --------------------------------------------------------------------------- models


class Message(Strict):
    """One chat message sent to a model provider."""

    role: Literal["system", "user", "assistant"]
    content: str


class Completion(Strict):
    """A model response with its token usage."""

    text: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)


class Usage(Strict):
    """Token, cost and call totals."""

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    model_calls: int = Field(default=0, ge=0)

    def plus(self, other: Usage) -> Usage:
        """Return the element-wise sum."""
        return Usage(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cost=self.cost + other.cost,
            model_calls=self.model_calls + other.model_calls,
        )

    @property
    def tokens(self) -> int:
        """Total tokens in and out."""
        return self.tokens_in + self.tokens_out


# --------------------------------------------------------------------------- tools & policy


class ToolSpec(Strict):
    """Descriptor of a tool, exportable as a function-calling style JSON Schema."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str
    args_schema: dict[str, Any]
    reversible: bool
    builtin: bool = True


class ToolResult(Strict):
    """Outcome of one tool call. ``output`` is redacted, truncated, untrusted data."""

    ok: bool
    exit_code: int | None = None
    output: str = ""
    truncated: bool = False


class Reversibility(StrEnum):
    """Policy classification of an action."""

    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class PolicyDecision(Strict):
    """Result of classifying an action."""

    action_name: str
    classification: Reversibility
    reason: str
    requires_human: bool = False

    @property
    def irreversible(self) -> bool:
        """True if the action is classified irreversible."""
        return self.classification is Reversibility.IRREVERSIBLE


# --------------------------------------------------------------------------- traces


class EscalationReason(StrEnum):
    """Why a tier did not accept a step."""

    NO_SKILL = "no_skill"
    GUARD_FALSE = "guard_false"
    DISAGREEMENT = "disagreement"
    PARSE_ERROR = "parse_error"
    ACCEPTANCE_FAILED = "acceptance_failed"
    IRREVERSIBLE_LOW_CONFIDENCE = "irreversible_low_confidence"
    IRREVERSIBLE_REQUIRES_HUMAN = "irreversible_requires_human"
    SKILL_UNSAFE = "skill_unsafe"
    TIER_ERROR = "tier_error"
    IN_DOUBT = "in_doubt"


class StepResult(Strict):
    """Recorded outcome of one executed step."""

    ok: bool
    output_hash: str
    checks_passed: bool | None = None
    exit_code: int | None = None
    summary: str = ""


class TraceStep(Strict):
    """One executed step as recorded in the append-only trace log."""

    type: Literal["step"] = "step"
    id: str
    run_id: str
    task_id: str
    attempt: int = Field(ge=1)
    index: int = Field(ge=0)
    ts: str
    situation: Situation
    action: Action
    result: StepResult
    route: str = Field(pattern=TIER_PATTERN)
    escalation_reason: str | None = None
    escalation_path: list[str] = Field(default_factory=list)
    skill_key: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    model_calls: int = Field(default=0, ge=0)
    verified: bool = False
    redacted: bool = False
    replayed: bool = False


class TraceVerdict(Strict):
    """Acceptance verdict for one task attempt; marks that attempt's steps verified or not."""

    type: Literal["verdict"] = "verdict"
    run_id: str
    task_id: str
    attempt: int = Field(ge=1)
    passed: bool
    ts: str
    step_ids: list[str] = Field(default_factory=list)


class TraceOverhead(Strict):
    """Usage spent at a step position that executed nothing (failed routing, ``done``,
    or a run interrupted by the budget or a human)."""

    type: Literal["overhead"] = "overhead"
    run_id: str
    task_id: str
    attempt: int = Field(ge=1)
    index: int = Field(ge=0)
    ts: str
    reason: str
    usage: Usage


class StepRequest(Strict):
    """Everything a ladder tier needs to propose the next action(s) for one step."""

    run_id: str
    task: Task
    situation: Situation
    attempt: int = Field(ge=1)
    context: str = ""
    planned_kinds: list[str] = Field(default_factory=list)
    open_mode: bool = False
    tools: list[str] = Field(default_factory=list)


class Proposal(Strict):
    """A tier's answer: one or more actions (skills may emit several) or ``done``."""

    tier: str = Field(pattern=TIER_PATTERN)
    actions: list[Action] = Field(default_factory=list)
    done: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    skill_key: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _actions_or_done(self) -> Proposal:
        if self.done == bool(self.actions):
            raise ValueError("a proposal has either actions or done, not both or neither")
        return self


class TierResult(Strict):
    """Outcome of asking one tier: a proposal or an escalation, plus the usage it cost."""

    tier: str = Field(pattern=TIER_PATTERN)
    proposal: Proposal | None = None
    escalation: EscalationReason | None = None
    usage: Usage = Field(default_factory=Usage)
    detail: str = ""

    @model_validator(mode="after")
    def _one_outcome(self) -> TierResult:
        if (self.proposal is None) == (self.escalation is None):
            raise ValueError("a tier result has exactly one of proposal or escalation")
        return self


class ApprovalRequest(Strict):
    """What a human is asked to approve (or to answer when no proposal exists)."""

    run_id: str
    task_id: str
    step_index: int = Field(ge=0)
    situation: Situation
    actions: list[Action] = Field(default_factory=list)
    decisions: list[PolicyDecision] = Field(default_factory=list)
    reason: str


class RouteDecision(Strict):
    """The router's accepted proposal for one step, with the path it took to get there."""

    route: str = Field(pattern=TIER_PATTERN)
    proposal: Proposal
    usage: Usage = Field(default_factory=Usage)
    escalation_reason: str | None = None
    escalation_path: list[str] = Field(default_factory=list)
    human_approved: bool = False


# --------------------------------------------------------------------------- skills


class SlotType(StrEnum):
    """Types a skill slot may take."""

    IDENTIFIER = "identifier"
    PATH = "path"
    STRING = "string"
    INT = "int"


class Slot(Strict):
    """A typed slot bound to a situation field."""

    type: SlotType
    source: str = Field(pattern=FIELD_PATTERN)


class GuardOp(StrEnum):
    """The only guard operators. There is no regex and no eval."""

    EQ = "eq"
    IN = "in"
    RANGE = "range"
    HAS_TYPE = "has_type"
    EXISTS = "exists"


class RangeValue(Strict):
    """Inclusive numeric range."""

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> RangeValue:
        if self.min > self.max:
            raise ValueError("range min must be <= max")
        return self


class Condition(Strict):
    """One guard condition: ``field op value``."""

    field: str = Field(pattern=FIELD_PATTERN)
    op: GuardOp
    value: Scalar | list[Scalar] | RangeValue | None = None

    @model_validator(mode="after")
    def _value_matches_op(self) -> Condition:
        value = self.value
        if self.op is GuardOp.EQ and not isinstance(value, str | int | float | bool):
            raise ValueError("eq needs a scalar value")
        if self.op is GuardOp.IN and (not isinstance(value, list) or not value):
            raise ValueError("in needs a non-empty list")
        if self.op is GuardOp.RANGE and not isinstance(value, RangeValue):
            raise ValueError("range needs {min, max}")
        if self.op is GuardOp.HAS_TYPE and value not in {t.value for t in SlotType}:
            raise ValueError("has_type needs a slot type")
        if self.op is GuardOp.EXISTS and value not in (None, True):
            raise ValueError("exists takes no value")
        return self


class Guard(Strict):
    """Conjunction of conditions. A skill fires only when every condition holds."""

    all: list[Condition] = Field(min_length=1)


class SkillStep(Strict):
    """A templated action. String args may contain ``{slot}`` or ``{slot|filter}``."""

    tool: str = Field(min_length=1, max_length=64)
    step_kind: str = Field(pattern=KIND_PATTERN)
    args: dict[str, ArgValue] = Field(default_factory=dict)


class Skill(Strict):
    """A declarative skill: data interpreted by the executor, never code."""

    schema_version: int = 1
    id: str = Field(pattern=r"^[a-z][a-z0-9\-]{2,63}$")
    version: int = Field(ge=1)
    task_kind: str = Field(pattern=KIND_PATTERN)
    slots: dict[str, Slot] = Field(default_factory=dict)
    guard: Guard
    steps: list[SkillStep] = Field(min_length=1)

    @property
    def key(self) -> str:
        """Registry key ``id@vN``."""
        return f"{self.id}@v{self.version}"


class SkillStatus(StrEnum):
    """Lifecycle status of a skill."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    DEMOTED = "demoted"


class HistoryEvent(Strict):
    """A lifecycle event with the evidence that justified it."""

    ts: str
    event: Literal["created", "imported", "promoted", "demoted"]
    reason: str
    forced: bool = False
    evidence: dict[str, Scalar] = Field(default_factory=dict)


class SkillManifest(Strict):
    """Registry record for one skill version."""

    id: str
    version: int
    content_hash: str
    guard_summary: str
    source_file: str
    provenance: list[str] = Field(default_factory=list)
    derived_runs: list[str] = Field(default_factory=list)
    origin: Literal["mined", "imported"] = "mined"
    pass_rate: float = 0.0
    shadow_runs: int = 0
    shadow_passes: int = 0
    unsafe_diffs: int = 0
    live_runs: int = 0
    live_passes: int = 0
    status: SkillStatus = SkillStatus.CANDIDATE
    created_at: str
    history: list[HistoryEvent] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """Registry key ``id@vN``."""
        return f"{self.id}@v{self.version}"


class RegistryManifest(Strict):
    """The on-disk registry index (``skills/manifest.json``)."""

    schema_version: int = 1
    registry_version: int = 0
    skills: list[SkillManifest] = Field(default_factory=list)


# --------------------------------------------------------------------------- state


class Cursor(Strict):
    """Where a run is: which run, whether it is unfinished, and the in-flight task attempt."""

    run_id: str | None = None
    active: bool = False
    task_id: str | None = None
    attempt: int = 0
    floor: int = 0
    failures_at_floor: int = 0


class Checkpoint(Strict):
    """Atomic snapshot of project state, verified by a SHA-256 over its canonical JSON."""

    schema_version: int = 1
    seq: int = Field(ge=1)
    created_at: str
    plan: Plan
    memory_snapshot_id: int = Field(ge=0)
    registry_version: int = Field(ge=0)
    cursor: Cursor
    hash: str = ""


class MemoryKind(StrEnum):
    """Kinds of memory entries."""

    DECISION = "decision"
    NOTE = "note"
    FACT = "fact"
    TASK = "task"


class MemoryEntry(Strict):
    """One project memory entry. Entries are archived, never deleted."""

    id: int
    kind: MemoryKind
    text: str
    tags: list[str] = Field(default_factory=list)
    source_task: str | None = None
    created_at: str
    archived: bool = False
    summary_of: list[int] = Field(default_factory=list)


class JournalStatus(StrEnum):
    """Write-ahead journal entry status."""

    STARTED = "started"
    COMPLETED = "completed"


class JournalEntry(Strict):
    """Idempotency record for one executed action."""

    key: str
    run_id: str
    task_id: str
    attempt: int
    step_index: int
    action_hash: str
    status: JournalStatus
    result: ToolResult | None = None
    updated_at: str


class Event(Frozen):
    """A redacted, immutable notification published on the event bus."""

    kind: str
    ts: str
    run_id: str | None = None
    task_id: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


class TaskStatus(Strict):
    """One task's status line."""

    id: str
    title: str
    kind: str
    state: TaskState
    attempts: int


class RunSummary(Strict):
    """Result of ``run`` or ``resume``."""

    run_id: str | None
    resumed: bool = False
    nothing_to_do: bool = False
    tasks: list[TaskStatus] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    steps: int = 0
    route_mix: dict[str, int] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    exit_code: int = 0


class StatusReport(Strict):
    """Result of ``status``."""

    workspace: str
    state_dir: str
    has_plan: bool
    goal: str | None = None
    run_id: str | None = None
    run_active: bool = False
    tasks: list[TaskStatus] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    skills: dict[str, int] = Field(default_factory=dict)
    runs: int = 0


class PreviewStep(Strict):
    """One line of a ``--dry-run`` preview."""

    task_id: str
    step_index: int
    step_kind: str
    route: str
    detail: str
    actions: list[Action] = Field(default_factory=list)


class RouteCost(Strict):
    """Totals for one route."""

    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    model_calls: int = 0


class CostReport(Strict):
    """Result of ``report``."""

    runs: int
    steps: int
    by_route: dict[str, RouteCost]
    overhead: Usage
    total: Usage
    skills: dict[str, int]
    baseline_cost: float
    estimated_saving: float
    saving_pct: float
    note: str


class InitReport(Strict):
    """Result of ``init``."""

    workspace: str
    state_dir: str
    config_path: str
    created: list[str] = Field(default_factory=list)
    dry_run: bool = False


class PlanReport(Strict):
    """Result of ``plan``."""

    plan: Plan | None = None
    usage: Usage = Field(default_factory=Usage)
    dry_run: bool = False
    detail: str = ""


class BenchRun(Strict):
    """Metrics of one benchmark run."""

    run: int
    run_id: str
    cost: float
    tokens_in: int
    tokens_out: int
    model_calls: int
    route_mix: dict[str, int]
    skills: dict[str, int]
    acceptance_pass_rate: float
    tasks: int
    steps: int
    mechanical_steps: int
    skill_steps: int
    exit_code: int


class BenchSummary(Strict):
    """Cross-run summary."""

    cost_first: float
    cost_last: float
    cost_ratio: float | None
    first_run_with_active_skill: int | None
    acceptance_pass_rate: float
    total_cost: float


class BenchReport(Strict):
    """Benchmark output."""

    scenario: str
    description: str
    seed: int
    profile: str | None
    runs: list[BenchRun] = Field(default_factory=list)
    summary: BenchSummary
    note: str = (
        "MockModel benchmark: proves routing, mining, shadow-testing and promotion mechanics; "
        "it does not prove cost savings with a real model."
    )


# --------------------------------------------------------------------------- schema export


def exported_models() -> dict[str, type[BaseModel]]:
    """Return the models whose JSON Schemas are committed under ``/schemas``."""
    return {
        "action": Action,
        "bench_report": BenchReport,
        "checkpoint": Checkpoint,
        "config": Config,
        "cost_report": CostReport,
        "event": Event,
        "memory_entry": MemoryEntry,
        "plan": Plan,
        "policy_decision": PolicyDecision,
        "registry_manifest": RegistryManifest,
        "situation": Situation,
        "skill": Skill,
        "skill_manifest": SkillManifest,
        "tool_spec": ToolSpec,
        "trace_overhead": TraceOverhead,
        "trace_step": TraceStep,
        "trace_verdict": TraceVerdict,
    }


def render_schema(model: type[BaseModel]) -> str:
    """Render a model's JSON Schema deterministically (sorted keys, 2-space indent)."""
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def export_schemas(directory: Path) -> list[Path]:
    """Write every exported schema into ``directory`` and return the written paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in exported_models().items():
        path = directory / f"{name}.schema.json"
        path.write_text(render_schema(model), encoding="utf-8")
        written.append(path)
    return written


def check_schemas(directory: Path) -> list[str]:
    """Return drift problems between committed schemas in ``directory`` and generated ones."""
    problems: list[str] = []
    expected = {f"{name}.schema.json": model for name, model in exported_models().items()}
    for filename, model in sorted(expected.items()):
        path = directory / filename
        if not path.is_file():
            problems.append(f"missing: {filename}")
        elif path.read_text(encoding="utf-8") != render_schema(model):
            problems.append(f"differs: {filename}")
    if directory.is_dir():
        for path in sorted(directory.glob("*.schema.json")):
            if path.name not in expected:
                problems.append(f"unexpected: {path.name}")
    return problems


def _matches(pattern: str, text: str) -> bool:
    return re.fullmatch(pattern, text) is not None
