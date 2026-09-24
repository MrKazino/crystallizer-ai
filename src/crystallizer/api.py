"""The stable programmatic API: :class:`Harness`.

Other workflows embed crystallizer through this facade; the CLI is a thin layer over it.

    with Harness.open("path/to/workspace", model=my_client, plugins=[MyPlugin()]) as harness:
        harness.plan("Add a parser module")
        summary = harness.run()

State is opened lazily, so ``dry_run=True`` never creates or writes anything, never takes the
lock, and never calls a model.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from crystallizer.approval import Approver, TTYApprover
from crystallizer.budget import BudgetGovernor
from crystallizer.checkpoint import CheckpointStore
from crystallizer.clock import Clock, SystemClock, make_rng
from crystallizer.config import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_CONFIG_TEXT,
    Config,
    load_config,
    resolve_state_dir,
)
from crystallizer.context import ContextBuilder
from crystallizer.db import Database
from crystallizer.errors import ConfigError, UsageError
from crystallizer.events import EventBus
from crystallizer.extensions import ModelFactory, Plugin, PluginContext, Registry, load_plugins
from crystallizer.faults import NO_FAULTS, FaultInjector
from crystallizer.journal import Journal
from crystallizer.lock import RunLock
from crystallizer.memory import Memory
from crystallizer.models import AnthropicClient, CostTable, MockModel, MockScript, ModelClient
from crystallizer.planner import Planner, TaskLedger, validate_plan
from crystallizer.policy import Policy
from crystallizer.redaction import Redactor, configure_default, load_env_values
from crystallizer.router import LadderRouter
from crystallizer.runner import Runner, RunnerDeps, preview_plan, task_statuses
from crystallizer.schemas import (
    CostReport,
    Cursor,
    InitReport,
    MemoryKind,
    Plan,
    PlanReport,
    PreviewStep,
    RunSummary,
    Skill,
    SkillManifest,
    SkillStatus,
    StatusReport,
    TaskState,
    ToolSpec,
)
from crystallizer.skills.registry import (
    SkillRegistry,
    load_manifest,
    read_skill,
    resolve_reference,
)
from crystallizer.tiers import (
    HumanTier,
    ModelTier,
    Proposer,
    SkillTier,
    TierContext,
    TierFactory,
)
from crystallizer.tools import Sandbox, ToolRegistry
from crystallizer.traces import TraceStore, cost_report

STATE_SUBDIRS = ("checkpoints", "traces", "skills/candidates", "skills/active", "skills/demoted")


@dataclass
class _State:
    db: Database
    memory: Memory
    journal: Journal
    ledger: TaskLedger


def load_mock_script(workspace: Path, relative: str) -> MockScript:
    """Load a MockModel script (JSON) named in ``[model] mock_script``; empty when unset."""
    if not relative:
        return MockScript()
    path = Path(relative)
    path = path if path.is_absolute() else workspace / path
    if not path.is_file():
        raise ConfigError(f"mock script not found: {relative}")
    return MockScript.model_validate_json(path.read_text(encoding="utf-8"))


class Harness:
    """Facade over configuration, state, plugins and the runner for one workspace."""

    def __init__(
        self,
        workspace: Path,
        config: Config,
        *,
        clock: Clock,
        rng: random.Random,
        model: ModelClient | None = None,
        approver: Approver | None = None,
        plugins: Sequence[Plugin] = (),
        faults: FaultInjector = NO_FAULTS,
        environ: Mapping[str, str] | None = None,
        dry_run: bool = False,
        seed: int = 0,
    ) -> None:
        """Assemble the harness; prefer :meth:`open`."""
        self.workspace = workspace.resolve()
        self.config = config
        self.clock = clock
        self.rng = rng
        self.dry_run = dry_run
        self.faults = faults
        self.approver: Approver = approver or TTYApprover()
        self.state_dir = resolve_state_dir(self.workspace, config)
        self.redactor = Redactor(
            load_env_values(self.workspace / ".env"), config.redaction.min_env_value_length
        )
        configure_default(self.redactor)
        self.bus = EventBus(clock, self.redactor)
        self.sandbox = Sandbox(self.workspace, self.state_dir, config.tools, self.redactor, environ)
        self.tools = ToolRegistry(self.sandbox)
        self.model_providers: Registry[ModelFactory] = Registry("model provider")
        self.model_providers.register(
            "mock",
            lambda cfg: MockModel(load_mock_script(self.workspace, cfg.model.mock_script), seed),
            builtin=True,
        )
        self.model_providers.register(
            "anthropic", lambda cfg: AnthropicClient(cfg.model, environ=environ), builtin=True
        )
        self.tier_factories: Registry[TierFactory] = Registry("tier")
        context = PluginContext(
            tools=self.tools, models=self.model_providers, tiers=self.tier_factories, bus=self.bus
        )
        self.plugins = load_plugins(config.plugins.enabled, plugins, context)
        plugin_tools = {spec.name: spec for spec in self.tools.specs() if not spec.builtin}
        self.policy = Policy(config.policy, plugin_tools)
        self.costs = CostTable(config.cost)
        self._model_override = model
        self._model: ModelClient | None = None
        self._state: _State | None = None
        self._registry: SkillRegistry | None = None
        self.checkpoints = CheckpointStore(self.state_dir, clock, self.redactor)
        self.traces = TraceStore(self.state_dir / "traces")

    @classmethod
    def open(
        cls,
        workspace: Path | str = ".",
        *,
        config_path: Path | None = None,
        profile: str | None = None,
        clock: Clock | None = None,
        seed: int = 0,
        model: ModelClient | None = None,
        approver: Approver | None = None,
        plugins: Sequence[Plugin] = (),
        faults: FaultInjector = NO_FAULTS,
        environ: Mapping[str, str] | None = None,
        dry_run: bool = False,
    ) -> Harness:
        """Load configuration for ``workspace`` and build a harness."""
        root = Path(workspace)
        if not root.is_dir():
            raise UsageError(f"workspace does not exist: {root}")
        config = load_config(root, config_path, profile)
        return cls(
            root,
            config,
            clock=clock or SystemClock(),
            rng=make_rng(seed),
            model=model,
            approver=approver,
            plugins=plugins,
            faults=faults,
            environ=environ,
            dry_run=dry_run,
            seed=seed,
        )

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> Harness:
        """Return self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close state."""
        self.close()

    def close(self) -> None:
        """Close the state database if it was opened."""
        if self._state is not None:
            self._state.db.close()
            self._state = None

    @property
    def model(self) -> ModelClient:
        """The configured model client (built on first use)."""
        if self._model_override is not None:
            return self._model_override
        if self._model is None:
            self._model = self.model_providers.get(self.config.model.provider)(self.config)
        return self._model

    def _open_state(self) -> _State:
        if self.dry_run:
            raise UsageError("internal error: state requested during a dry run")
        if self._state is None:
            db = Database(self.state_dir / "state.db")
            self._state = _State(
                db=db,
                memory=Memory(db, self.clock, self.redactor, self.config.memory.half_life_days),
                journal=Journal(db, self.clock, self.faults),
                ledger=TaskLedger(db, self.clock),
            )
        return self._state

    def registry(self) -> SkillRegistry:
        """The skill registry (opens state)."""
        state = self._open_state()
        if self._registry is None:
            self._registry = SkillRegistry(
                self.state_dir, state.db, self.config.skills, self.clock, self.policy, self.bus
            )
        return self._registry

    def _lock(self) -> RunLock:
        return RunLock(self.state_dir)

    # ------------------------------------------------------------------ commands

    def init(self) -> InitReport:
        """Create the default config file (if absent) and the state directory layout."""
        config_path = self.workspace / DEFAULT_CONFIG_NAME
        planned: list[str] = []
        if not config_path.exists():
            planned.append(DEFAULT_CONFIG_NAME)
        for sub in STATE_SUBDIRS:
            if not (self.state_dir / sub).is_dir():
                planned.append(str((self.state_dir / sub).relative_to(self.state_dir.parent)))
        report = InitReport(
            workspace=str(self.workspace),
            state_dir=str(self.state_dir),
            config_path=str(config_path),
            created=planned,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            return report
        if not config_path.exists():
            config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
        for sub in STATE_SUBDIRS:
            (self.state_dir / sub).mkdir(parents=True, exist_ok=True)
        self._open_state()
        return report

    def plan(self, goal: str) -> PlanReport:
        """Ask the planner for a plan and store it as the first checkpoint of a new project."""
        tier = self.config.model.planner_tier
        if self.dry_run:
            return PlanReport(dry_run=True, detail=f"would call {tier} model to plan: {goal}")
        with self._lock():
            latest = self.checkpoints.latest()
            if latest is not None and latest.cursor.active:
                raise UsageError(
                    f"run {latest.cursor.run_id} is unfinished: resume it before planning again"
                )
            state = self._open_state()
            planner = Planner(
                self.model,
                self.costs,
                self.redactor,
                tier=tier,
                max_tokens=self.config.model.max_tokens,
                budget=BudgetGovernor(self.config.budget),
            )
            plan, usage = planner.plan(goal)
            self.checkpoints.save(plan, Cursor(), state.memory.max_id(), 0)
            state.memory.add(
                MemoryKind.DECISION,
                f"planned goal with {len(plan.tasks)} task(s): {plan.goal}",
                ["plan"],
            )
        return PlanReport(plan=plan, usage=usage)

    def adopt_plan(self, plan: Plan) -> Plan:
        """Use an externally produced plan (another planner, a workflow engine, a scenario).

        The plan is validated as a DAG, every task is reset to pending, and it is stored as the
        new project plan, exactly as if the planner had produced it.
        """
        fresh = validate_plan(
            plan.model_copy(
                update={
                    "tasks": [
                        task.model_copy(update={"state": TaskState.PENDING, "attempts": 0})
                        for task in plan.tasks
                    ]
                }
            )
        )
        if self.dry_run:
            return fresh
        with self._lock():
            latest = self.checkpoints.latest()
            if latest is not None and latest.cursor.active:
                raise UsageError(
                    f"run {latest.cursor.run_id} is unfinished: resume it before adopting a plan"
                )
            state = self._open_state()
            self.checkpoints.save(fresh, Cursor(), state.memory.max_id(), self.registry().version)
            state.memory.add(
                MemoryKind.DECISION,
                f"adopted plan with {len(fresh.tasks)} task(s): {fresh.goal}",
                ["plan"],
            )
        return fresh

    def _runner(self) -> Runner:
        state = self._open_state()
        budget = BudgetGovernor(self.config.budget)
        context = ContextBuilder(
            self.sandbox,
            state.memory,
            self.redactor,
            budget_tokens=self.config.context.budget_tokens,
            chars_per_token=self.config.context.chars_per_token,
            memory_items=self.config.context.memory_items,
            max_file_bytes=self.config.tools.max_file_bytes,
        )
        router = self._build_router(budget)
        deps = RunnerDeps(
            config=self.config,
            clock=self.clock,
            rng=self.rng,
            checkpoints=self.checkpoints,
            ledger=state.ledger,
            memory=state.memory,
            journal=state.journal,
            traces_dir=self.state_dir / "traces",
            tools=self.tools,
            policy=self.policy,
            router=router,
            context=context,
            budget=budget,
            bus=self.bus,
            approver=self.approver,
            redactor=self.redactor,
            registry=self.registry(),
            faults=self.faults,
        )
        return Runner(deps)

    def _build_router(self, budget: BudgetGovernor) -> LadderRouter:
        """Assemble the configured ladder: built-in and plugin tiers, then the human."""
        router_config = self.config.router
        registry = self.registry()
        tiers: list[Proposer] = []
        human: HumanTier | None = None
        for name in router_config.ladder:
            if name == "skill":
                tiers.append(SkillTier(registry.active_skills, lambda: registry.version))
            elif name in ("small", "large"):
                samples = router_config.samples if name == "small" else router_config.large_samples
                tiers.append(
                    ModelTier(
                        name,
                        name,
                        self.model,
                        self.costs,
                        budget,
                        samples=samples,
                        agreement=router_config.agreement,
                        max_tokens=self.config.model.max_tokens,
                    )
                )
            elif name == "human":
                human = HumanTier(self.approver, self.policy)
            else:
                context = TierContext(
                    config=self.config,
                    model=self.model,
                    costs=self.costs,
                    budget=budget,
                    policy=self.policy,
                    bus=self.bus,
                    tools=tuple(self.tools.specs()),
                )
                tier = self.tier_factories.get(name)(context)
                if tier.name != name:
                    raise ConfigError(f"tier registered as {name!r} reports name {tier.name!r}")
                tiers.append(tier)

        def demote_unsafe(skill_key: str, reason: str) -> None:
            registry.demote_unsafe(skill_key, reason)

        return LadderRouter(
            tiers,
            human=human,
            policy=self.policy,
            irreversible_confidence=router_config.irreversible_confidence,
            on_unsafe_skill=demote_unsafe,
        )

    def run(self) -> RunSummary:
        """Start a new run over the unfinished tasks of the current plan."""
        with self._lock():
            return self._runner().run(resume=False)

    def resume(self) -> RunSummary:
        """Continue the unfinished run, never repeating a completed side effect."""
        with self._lock():
            return self._runner().run(resume=True)

    def preview(self) -> list[PreviewStep]:
        """Dry-run routing preview; reads only the latest checkpoint."""
        checkpoint = self.checkpoints.latest()
        if checkpoint is None:
            raise UsageError("no plan found: run `crystallizer plan GOAL` first")
        active: list[tuple[Skill, SkillManifest]] = [
            (read_skill(self.skills_dir, entry), entry)
            for entry in load_manifest(self.skills_dir).skills
            if entry.status is SkillStatus.ACTIVE
        ]
        active.sort(
            key=lambda p: (-len(p[0].guard.all), -len(p[0].steps), -p[1].pass_rate, p[1].key)
        )
        ladder = self.config.router.ladder
        model_tiers = [name for name in ladder if name not in ("skill", "human")]
        return preview_plan(
            checkpoint.plan,
            active,
            model_tier=model_tiers[0] if model_tiers else "human",
            has_skill_tier="skill" in ladder,
        )

    def report(self) -> CostReport:
        """Cost by route, skill counts, and the estimated saving against an all-large baseline."""
        counts = {status.value: 0 for status in SkillStatus}
        for entry in self.skills_list():
            counts[entry.status.value] += 1
        return cost_report(
            self.traces.load_all(),
            self.traces.overheads(),
            self.costs,
            counts,
            len(self.traces.run_ids()),
        )

    def status(self) -> StatusReport:
        """Read-only summary of the plan, the run cursor and recorded runs."""
        checkpoint = self.checkpoints.latest()
        counts_by_status: dict[str, int] = {}
        for entry in self.skills_list():
            counts_by_status[entry.status.value] = counts_by_status.get(entry.status.value, 0) + 1
        report = StatusReport(
            workspace=str(self.workspace),
            state_dir=str(self.state_dir),
            has_plan=checkpoint is not None,
            runs=len(self.traces.run_ids()),
            skills=dict(sorted(counts_by_status.items())),
        )
        if checkpoint is None:
            return report
        tasks = task_statuses(checkpoint.plan)
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.state.value] = counts.get(task.state.value, 0) + 1
        return report.model_copy(
            update={
                "goal": checkpoint.plan.goal,
                "run_id": checkpoint.cursor.run_id,
                "run_active": checkpoint.cursor.active,
                "tasks": tasks,
                "counts": dict(sorted(counts.items())),
            }
        )

    # ------------------------------------------------------------------ skills

    @property
    def skills_dir(self) -> Path:
        """``state_dir/skills``."""
        return self.state_dir / "skills"

    def skills_list(self) -> list[SkillManifest]:
        """Every skill version in the registry (read-only)."""
        return load_manifest(self.skills_dir).skills

    def skill_show(self, reference: str) -> dict[str, object]:
        """Manifest entry and skill JSON of ``reference`` (read-only)."""
        entry = resolve_reference(load_manifest(self.skills_dir), reference)
        skill = read_skill(self.skills_dir, entry)
        return {"manifest": entry.model_dump(mode="json"), "skill": skill.model_dump(mode="json")}

    def skill_export(self, reference: str) -> dict[str, object]:
        """Skill JSON of ``reference`` for use in another workspace (read-only)."""
        entry = resolve_reference(load_manifest(self.skills_dir), reference)
        return read_skill(self.skills_dir, entry).model_dump(mode="json")

    def skill_promote(
        self, reference: str, *, force: bool = False, reason: str | None = None
    ) -> SkillManifest:
        """Promote a candidate (same checks as automatic promotion; ``force`` needs a reason)."""
        with self._lock():
            return self.registry().promote(reference, force=force, reason=reason)

    def skill_demote(self, reference: str, *, reason: str) -> SkillManifest:
        """Demote an active skill."""
        with self._lock():
            return self.registry().demote(reference, reason=reason)

    def skills_mine(self) -> list[SkillManifest]:
        """Mine recorded traces into new candidates."""
        with self._lock():
            return self.registry().mine(self.traces)

    def skills_evaluate(self) -> int:
        """Shadow-evaluate candidates over every recorded verified attempt."""
        with self._lock():
            return self.registry().evaluate(self.traces)

    def skill_import(self, path: Path) -> SkillManifest | None:
        """Import skill JSON as a candidate (it must pass local shadow testing)."""
        text = path.read_text(encoding="utf-8")
        with self._lock():
            return self.registry().import_skill(text)

    def tool_specs(self) -> list[ToolSpec]:
        """Descriptors of every available tool (JSON Schema arguments)."""
        return self.tools.specs()
