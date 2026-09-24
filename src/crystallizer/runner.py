"""The runner: plan → context → route → act (journaled, sandboxed) → trace → verify → learn.

One runner serves every phase. For each ready task it builds budgeted context, asks the
:class:`~crystallizer.router.StepRouter` for each step, enforces the irreversible-action policy once
more before executing (defense in depth), executes through the write-ahead journal, records
traces, runs acceptance commands through the same sandbox and policy, records the verdict, feeds
the skill registry (live accounting, shadow evaluation), and checkpoints at run start, at every
attempt start and after every task. At the end of a run it mines new candidates and promotes
candidates whose shadow evidence meets the criteria.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from crystallizer.approval import Approver
from crystallizer.budget import BudgetGovernor
from crystallizer.checkpoint import CheckpointStore
from crystallizer.clock import Clock, iso, new_run_id
from crystallizer.config import Config
from crystallizer.context import ContextBuilder
from crystallizer.errors import (
    BudgetExhaustedError,
    CrystallizerError,
    HumanApprovalError,
    InDoubtActionError,
    UsageError,
)
from crystallizer.events import EventBus, EventKind
from crystallizer.faults import NO_FAULTS, FaultInjector
from crystallizer.hashing import sha256_hex
from crystallizer.journal import Journal, JournalOutcome
from crystallizer.logging_setup import get_logger, log_context
from crystallizer.memory import Memory
from crystallizer.planner import TaskLedger, replace_task, topological_order
from crystallizer.policy import Policy
from crystallizer.redaction import Redactor
from crystallizer.router import NoProposal, StepRouter
from crystallizer.schemas import (
    Action,
    ApprovalRequest,
    Checkpoint,
    Cursor,
    MemoryKind,
    Plan,
    PreviewStep,
    Proposal,
    RouteDecision,
    RunSummary,
    Situation,
    Skill,
    SkillManifest,
    StepRequest,
    StepResult,
    Task,
    TaskState,
    TaskStatus,
    ToolResult,
    TraceOverhead,
    TraceStep,
    TraceVerdict,
    Usage,
)
from crystallizer.skills.registry import SkillRegistry
from crystallizer.tiers import find_skill
from crystallizer.tools import ToolRegistry
from crystallizer.traces import TraceRecorder, TraceStore

OPEN_STEP_KIND = "open"
_log = get_logger("runner")


@dataclass
class RunnerDeps:
    """Everything the runner needs, injected so tests control every dependency."""

    config: Config
    clock: Clock
    rng: random.Random
    checkpoints: CheckpointStore
    ledger: TaskLedger
    memory: Memory
    journal: Journal
    traces_dir: Path
    tools: ToolRegistry
    policy: Policy
    router: StepRouter
    context: ContextBuilder
    budget: BudgetGovernor
    bus: EventBus
    approver: Approver
    redactor: Redactor
    registry: SkillRegistry | None = None
    faults: FaultInjector = field(default=NO_FAULTS)


@dataclass
class AttemptResult:
    """Outcome of one task attempt."""

    passed: bool
    steps: list[TraceStep]
    reason: str


def task_statuses(plan: Plan) -> list[TaskStatus]:
    """Status lines for every task in plan order."""
    return [
        TaskStatus(id=t.id, title=t.title, kind=t.kind, state=t.state, attempts=t.attempts)
        for t in plan.tasks
    ]


def situation_for(task: Task, index: int) -> Situation:
    """The situation at step ``index`` of ``task``."""
    kind = task.steps[index].kind if index < len(task.steps) else OPEN_STEP_KIND
    return Situation(task_kind=task.kind, step_kind=kind, step_index=index, params=task.params)


class Runner:
    """Executes a plan's tasks in deterministic topological order."""

    def __init__(self, deps: RunnerDeps) -> None:
        """Bind dependencies."""
        self.deps = deps
        self._store = TraceStore(deps.traces_dir)
        self._recorded: dict[tuple[str, int, int], TraceStep] = {}

    # ------------------------------------------------------------------ checkpoints

    def _registry_version(self) -> int:
        return self.deps.registry.version if self.deps.registry is not None else 0

    def _save(self, plan: Plan, cursor: Cursor) -> Checkpoint:
        checkpoint = self.deps.checkpoints.save(
            plan, cursor, self.deps.memory.max_id(), self._registry_version()
        )
        self.deps.bus.publish(EventKind.CHECKPOINT_SAVED, run_id=cursor.run_id, seq=checkpoint.seq)
        return checkpoint

    def _load(self) -> Checkpoint:
        checkpoint = self.deps.checkpoints.latest()
        if checkpoint is None:
            raise UsageError("no plan found: run `crystallizer plan GOAL` first")
        return checkpoint

    # ------------------------------------------------------------------ run

    def run(self, *, resume: bool) -> RunSummary:
        """Start a new run (``resume=False``) or continue the unfinished one."""
        checkpoint = self._load()
        plan, cursor = checkpoint.plan, checkpoint.cursor
        ledger = self.deps.ledger
        if resume:
            if not cursor.active or cursor.run_id is None:
                return RunSummary(
                    run_id=cursor.run_id,
                    resumed=True,
                    nothing_to_do=True,
                    tasks=task_statuses(plan),
                )
            run_id = cursor.run_id
            self.deps.bus.publish(EventKind.RUN_STARTED, run_id=run_id, resumed=True)
            for task in plan.tasks:
                if task.state is TaskState.RUNNING:
                    plan = ledger.transition(
                        plan, task.id, TaskState.PENDING, run_id=run_id, reason="resume"
                    )
            self.deps.budget.charge(self._store.run_usage(run_id))
            # Steps already executed in this run are replayed from the trace (no model call)
            # and from the journal (no tool call). Redacted steps must be routed again.
            self._recorded = {
                (step.task_id, step.attempt, step.index): step
                for step in self._store.load_run(run_id)
                if not step.redacted
            }
        else:
            if cursor.active:
                raise UsageError(
                    f"run {cursor.run_id} is unfinished: use `crystallizer resume` to continue it"
                )
            run_id = new_run_id(self.deps.clock, self.deps.rng)
            while (self.deps.traces_dir / f"{run_id}.jsonl").exists():
                run_id = new_run_id(self.deps.clock, self.deps.rng)
            self.deps.bus.publish(EventKind.RUN_STARTED, run_id=run_id, resumed=False)
            for task in plan.tasks:
                if task.state in (TaskState.FAILED, TaskState.BLOCKED):
                    plan = ledger.transition(
                        plan, task.id, TaskState.PENDING, run_id=run_id, reason="new run"
                    )
            cursor = Cursor(run_id=run_id, active=True)
            self._save(plan, cursor)
        recorder = TraceRecorder(self.deps.traces_dir, run_id, self.deps.redactor)
        with log_context(run_id=run_id):
            try:
                for ordered in topological_order(plan.tasks):
                    plan, cursor = self._visit(plan, cursor, ordered.id, run_id, recorder)
            except (BudgetExhaustedError, HumanApprovalError) as exc:
                kind = (
                    EventKind.BUDGET_EXHAUSTED
                    if isinstance(exc, BudgetExhaustedError)
                    else EventKind.ESCALATION
                )
                self.deps.bus.publish(kind, run_id=run_id, reason=exc.message)
                self._save(plan, cursor)
                raise
            self._learn_from_run()
            cursor = Cursor(run_id=run_id, active=False)
            self._save(plan, cursor)
        return self._summary(plan, run_id, resume)

    def _learn_from_run(self) -> None:
        registry = self.deps.registry
        if registry is None:
            return
        if self.deps.config.skills.auto_mine:
            registry.mine(self._store)
        if self.deps.config.skills.auto_promote:
            registry.try_promote_all()

    def _summary(self, plan: Plan, run_id: str, resumed: bool) -> RunSummary:
        steps = self._store.load_run(run_id)
        mix = Counter(step.route for step in steps if not step.replayed)
        failed = [t.id for t in plan.tasks if t.state is TaskState.FAILED]
        blocked = [t.id for t in plan.tasks if t.state is TaskState.BLOCKED]
        summary = RunSummary(
            run_id=run_id,
            resumed=resumed,
            tasks=task_statuses(plan),
            failed=failed,
            blocked=blocked,
            steps=sum(mix.values()),
            route_mix=dict(sorted(mix.items())),
            usage=self._store.run_usage(run_id),
            exit_code=3 if failed or blocked else 0,
        )
        self.deps.bus.publish(EventKind.RUN_FINISHED, run_id=run_id, failed=failed, blocked=blocked)
        return summary

    # ------------------------------------------------------------------ tasks

    def _visit(
        self, plan: Plan, cursor: Cursor, task_id: str, run_id: str, recorder: TraceRecorder
    ) -> tuple[Plan, Cursor]:
        task = plan.task(task_id)
        if task.state is TaskState.DONE:
            return plan, cursor
        broken = [
            dep
            for dep in task.depends_on
            if plan.task(dep).state in (TaskState.FAILED, TaskState.BLOCKED)
        ]
        if broken:
            if task.state is not TaskState.BLOCKED:
                plan = self.deps.ledger.transition(
                    plan,
                    task_id,
                    TaskState.BLOCKED,
                    run_id=run_id,
                    reason=f"dependency failed: {', '.join(broken)}",
                )
                self._save(plan, cursor)
            return plan, cursor
        return self._run_task(plan, cursor, task_id, run_id, recorder)

    def _run_task(
        self, plan: Plan, cursor: Cursor, task_id: str, run_id: str, recorder: TraceRecorder
    ) -> tuple[Plan, Cursor]:
        deps = self.deps
        plan = deps.ledger.transition(
            plan, task_id, TaskState.RUNNING, run_id=run_id, reason="start"
        )
        resuming = cursor.task_id == task_id and cursor.attempt > 0
        attempt = cursor.attempt if resuming else plan.task(task_id).attempts + 1
        floor = cursor.floor if resuming else 0
        failures = cursor.failures_at_floor if resuming else 0
        with log_context(run_id=run_id, task_id=task_id):
            while True:
                plan = replace_task(
                    plan, plan.task(task_id).model_copy(update={"attempts": attempt})
                )
                cursor = Cursor(
                    run_id=run_id,
                    active=True,
                    task_id=task_id,
                    attempt=attempt,
                    floor=floor,
                    failures_at_floor=failures,
                )
                self._save(plan, cursor)
                deps.bus.publish(
                    EventKind.TASK_STARTED,
                    run_id=run_id,
                    task_id=task_id,
                    attempt=attempt,
                    floor=floor,
                )
                result = self._attempt(plan.task(task_id), run_id, attempt, floor, recorder)
                recorder.record_verdict(
                    TraceVerdict(
                        run_id=run_id,
                        task_id=task_id,
                        attempt=attempt,
                        passed=result.passed,
                        ts=iso(deps.clock.now()),
                        step_ids=[step.id for step in result.steps],
                    )
                )
                self._learn_from_attempt(run_id, task_id, attempt, result)
                if result.passed:
                    plan = deps.ledger.transition(
                        plan, task_id, TaskState.DONE, run_id=run_id, reason="acceptance passed"
                    )
                    self._remember(plan.task(task_id), "done", attempt)
                    break
                failures += 1
                if failures > deps.config.router.retries:
                    floor += 1
                    failures = 0
                if floor > deps.router.max_floor:
                    plan = deps.ledger.transition(
                        plan, task_id, TaskState.FAILED, run_id=run_id, reason=result.reason
                    )
                    self._remember(plan.task(task_id), f"failed: {result.reason}", attempt)
                    break
                attempt += 1
        cursor = Cursor(run_id=run_id, active=True)
        deps.faults.hit("runner.before_checkpoint")
        self._save(plan, cursor)
        deps.bus.publish(
            EventKind.TASK_FINISHED,
            run_id=run_id,
            task_id=task_id,
            state=plan.task(task_id).state.value,
            attempts=attempt,
        )
        return plan, cursor

    def _learn_from_attempt(
        self, run_id: str, task_id: str, attempt: int, result: AttemptResult
    ) -> None:
        """Live accounting for skills used, shadow evaluation of candidates on verified work."""
        registry = self.deps.registry
        if registry is None:
            return
        occurrence = f"{run_id}:{task_id}:{attempt}"
        used = sorted(
            {s.skill_key for s in result.steps if s.route == "skill" and s.skill_key is not None}
        )
        for skill_key in used:
            registry.record_live(skill_key, occurrence, result.passed)
        if result.passed:
            registry.observe([step.model_copy(update={"verified": True}) for step in result.steps])

    def _remember(self, task: Task, outcome: str, attempts: int) -> None:
        text = f"task {task.id} ({task.kind}) {outcome} after {attempts} attempt(s): {task.title}"
        existing = [
            e
            for e in self.deps.memory.entries()
            if e.source_task == task.id and e.kind is MemoryKind.TASK and e.text == text
        ]
        if not existing:
            self.deps.memory.add(MemoryKind.TASK, text, ["task", task.kind], task.id)

    # ------------------------------------------------------------------ attempts

    def _overhead(
        self,
        recorder: TraceRecorder,
        task: Task,
        attempt: int,
        index: int,
        usage: Usage,
        reason: str,
    ) -> None:
        recorder.record_overhead(
            TraceOverhead(
                run_id=recorder.path.stem,
                task_id=task.id,
                attempt=attempt,
                index=index,
                ts=iso(self.deps.clock.now()),
                reason=reason,
                usage=usage,
            )
        )

    def _route(
        self, request: StepRequest, floor: int, recorder: TraceRecorder
    ) -> RouteDecision | NoProposal:
        task, attempt, index = request.task, request.attempt, request.situation.step_index
        replayed = self._replayed(task.id, attempt, index)
        if replayed is not None:
            return replayed
        try:
            decision = self.deps.router.route(request, floor)
        except NoProposal as exc:
            self._overhead(recorder, task, attempt, index, exc.usage, f"no proposal: {exc}")
            return exc
        except CrystallizerError as exc:
            if exc.usage is not None:
                self._overhead(recorder, task, attempt, index, exc.usage, exc.message)
            raise
        _log.info(
            "step routed",
            extra={
                "step_index": index,
                "route": decision.route,
                "escalation_reason": decision.escalation_reason,
                "escalation_path": decision.escalation_path,
                "tokens_in": decision.usage.tokens_in,
                "tokens_out": decision.usage.tokens_out,
                "cost": decision.usage.cost,
            },
        )
        if decision.escalation_path:
            self.deps.bus.publish(
                EventKind.ESCALATION,
                run_id=request.run_id,
                task_id=task.id,
                step_index=index,
                path=decision.escalation_path,
                route=decision.route,
            )
        return decision

    def _attempt(
        self, task: Task, run_id: str, attempt: int, floor: int, recorder: TraceRecorder
    ) -> AttemptResult:
        deps = self.deps
        open_mode = not task.steps
        limit = deps.config.runner.max_open_steps if open_mode else len(task.steps)
        executed: list[TraceStep] = []
        digest: list[str] = []
        index = 0
        tool_names = [spec.name for spec in deps.tools.specs()]
        while index < limit:
            recent = list(reversed(digest[-deps.config.context.trace_items :]))
            context = deps.context.build(task, trace_digest=recent)
            request = StepRequest(
                run_id=run_id,
                task=task,
                situation=situation_for(task, index),
                attempt=attempt,
                context=context.render(),
                planned_kinds=[step.kind for step in task.steps[index:]],
                open_mode=open_mode,
                tools=tool_names,
            )
            decision = self._route(request, floor, recorder)
            if isinstance(decision, NoProposal):
                return AttemptResult(False, executed, f"no proposal ({decision})")
            deps.bus.publish(
                EventKind.STEP_ROUTED,
                run_id=run_id,
                task_id=task.id,
                step_index=index,
                route=decision.route,
                escalation_path=decision.escalation_path,
                cost=decision.usage.cost,
            )
            proposal = decision.proposal
            if proposal.done:
                self._overhead(recorder, task, attempt, index, decision.usage, "done")
                if open_mode:
                    break
                return AttemptResult(False, executed, "done proposed at a planned step")
            if not open_mode and index + len(proposal.actions) > limit:
                self._overhead(recorder, task, attempt, index, decision.usage, "too many actions")
                return AttemptResult(False, executed, "proposal exceeds planned steps")
            for offset, action in enumerate(proposal.actions):
                step_index = index + offset
                usage = decision.usage if offset == 0 else Usage()
                step = self._execute(
                    task, run_id, attempt, step_index, action, decision, usage, recorder
                )
                executed.append(step)
                digest.append(
                    f"step {step_index} {action.tool} -> {'ok' if step.result.ok else 'failed'}"
                )
                deps.faults.hit("runner.after_step")
                if not step.result.ok:
                    return AttemptResult(
                        False, executed, f"step {step_index} ({action.tool}) failed"
                    )
            index += len(proposal.actions)
        if not self._acceptance(task, run_id, attempt):
            return AttemptResult(False, executed, "acceptance commands failed")
        return AttemptResult(True, executed, "acceptance passed")

    def _replayed(self, task_id: str, attempt: int, index: int) -> RouteDecision | None:
        recorded = self._recorded.get((task_id, attempt, index))
        if recorded is None:
            return None
        return RouteDecision(
            route=recorded.route,
            proposal=Proposal(
                tier=recorded.route,
                actions=[recorded.action],
                confidence=1.0,
                skill_key=recorded.skill_key,
            ),
            escalation_reason=recorded.escalation_reason,
            escalation_path=recorded.escalation_path,
            human_approved=True,
        )

    def _gate(
        self,
        task: Task,
        run_id: str,
        step_index: int,
        actions: list[Action],
        reason: str,
        *,
        force: bool = False,
    ) -> None:
        """Require human approval for irreversible, non-allow-listed actions (fail closed)."""
        decisions = [self.deps.policy.classify(action) for action in actions]
        if not force and not any(decision.requires_human for decision in decisions):
            return
        request = ApprovalRequest(
            run_id=run_id,
            task_id=task.id,
            step_index=step_index,
            situation=situation_for(task, step_index),
            actions=actions,
            decisions=decisions,
            reason=reason,
        )
        if not self.deps.approver.approve(request):
            raise HumanApprovalError(f"human denied: {reason}")

    def _execute(
        self,
        task: Task,
        run_id: str,
        attempt: int,
        step_index: int,
        action: Action,
        decision: RouteDecision,
        usage: Usage,
        recorder: TraceRecorder,
    ) -> TraceStep:
        deps = self.deps
        if not decision.human_approved:
            self._gate(task, run_id, step_index, [action], "irreversible action")

        def run_tool(target: Action = action) -> ToolResult:
            return deps.tools.execute(target)

        def journaled(allow_redo: bool) -> JournalOutcome:
            return deps.journal.execute(
                run_id=run_id,
                task_id=task.id,
                attempt=attempt,
                step_index=step_index,
                action=action,
                run=run_tool,
                allow_redo=allow_redo,
            )

        try:
            outcome = journaled(deps.policy.is_idempotent(action))
        except InDoubtActionError as exc:
            self._gate(task, run_id, step_index, [action], exc.message, force=True)
            outcome = journaled(True)
        result = outcome.result
        step = TraceStep(
            id=f"{run_id}:{task.id}:{attempt}:{step_index}",
            run_id=run_id,
            task_id=task.id,
            attempt=attempt,
            index=step_index,
            ts=iso(deps.clock.now()),
            situation=situation_for(task, step_index),
            action=action,
            result=StepResult(
                ok=result.ok,
                output_hash=sha256_hex(result.output),
                exit_code=result.exit_code,
                summary=result.output[:200],
            ),
            route=decision.route,
            escalation_reason=decision.escalation_reason,
            escalation_path=decision.escalation_path,
            skill_key=decision.proposal.skill_key,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost=usage.cost,
            model_calls=usage.model_calls,
            replayed=outcome.replayed,
        )
        stored = recorder.record_step(step)
        deps.bus.publish(
            EventKind.STEP_EXECUTED,
            run_id=run_id,
            task_id=task.id,
            step_index=step_index,
            tool=action.tool,
            ok=result.ok,
            route=decision.route,
            replayed=outcome.replayed,
        )
        return stored

    def _acceptance(self, task: Task, run_id: str, attempt: int) -> bool:
        deps = self.deps
        last = max(len(task.steps) - 1, 0)
        for argv in task.acceptance_commands:
            action = Action(tool="shell", args={"argv": argv})
            self._gate(task, run_id, last, [action], "irreversible acceptance command")
            result = deps.tools.execute(action)
            if not result.ok:
                _log.info(
                    "acceptance command failed",
                    extra={"run": run_id, "task": task.id, "attempt": attempt, "argv": argv},
                )
                return False
        deps.faults.hit("runner.after_acceptance")
        return True


def preview_plan(
    plan: Plan,
    active: Sequence[tuple[Skill, SkillManifest]] = (),
    model_tier: str = "small",
    has_skill_tier: bool = True,
) -> list[PreviewStep]:
    """Dry-run: planned routing for every unfinished task. No tools, state or model calls."""
    lines: list[PreviewStep] = []
    for task in topological_order(plan.tasks):
        if task.state is TaskState.DONE:
            continue
        count = len(task.steps) or 1
        index = 0
        while index < count:
            situation = situation_for(task, index)
            planned = [step.kind for step in task.steps[index:]]
            match = None
            if has_skill_tier:
                match, _ = find_skill(active, situation, planned, open_mode=not task.steps)
            if match is None:
                lines.append(
                    PreviewStep(
                        task_id=task.id,
                        step_index=index,
                        step_kind=situation.step_kind,
                        route=model_tier,
                        detail=f"would call {model_tier} model",
                    )
                )
                index += 1
                continue
            _, entry, actions = match
            for offset, action in enumerate(actions):
                lines.append(
                    PreviewStep(
                        task_id=task.id,
                        step_index=index + offset,
                        step_kind=situation_for(task, index + offset).step_kind,
                        route="skill",
                        detail=f"skill {entry.key}",
                        actions=[action],
                    )
                )
            index += len(actions)
    return lines
