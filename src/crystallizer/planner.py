"""Goal to task DAG, with validation, deterministic ordering and persisted state transitions.

The model is asked for one JSON object ``{"tasks": [...]}``. The first JSON object in the reply
is validated with pydantic; every task starts ``pending``. Duplicate ids, unknown dependencies and
cycles are rejected. Topological order breaks ties by task id, so it is deterministic.

Task states: pending, running, done, failed, blocked. Every transition is validated against
:data:`ALLOWED_TRANSITIONS` and appended to the ``task_events`` table.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from pydantic import ValidationError

from crystallizer.budget import BudgetGovernor
from crystallizer.clock import Clock, iso
from crystallizer.db import Database
from crystallizer.errors import PlanError
from crystallizer.models import CostTable, ModelClient, encode_request, extract_json_object
from crystallizer.redaction import Redactor
from crystallizer.schemas import Message, Plan, Task, TaskState, Usage

ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.RUNNING, TaskState.BLOCKED}),
    TaskState.RUNNING: frozenset({TaskState.DONE, TaskState.FAILED, TaskState.PENDING}),
    TaskState.FAILED: frozenset({TaskState.PENDING}),
    TaskState.BLOCKED: frozenset({TaskState.PENDING}),
    TaskState.DONE: frozenset(),
}

PLANNER_SYSTEM_PROMPT = """\
You are the planner of an agent harness. Decompose the goal into a small DAG of tasks.
Reply with exactly one JSON object: {"tasks": [TASK, ...]} where TASK is
{"id": str, "title": str, "kind": str, "params": {name: scalar}, "depends_on": [task ids],
 "steps": [{"kind": str, "description": str}],
 "acceptance_commands": [[argv...]]}.
Acceptance commands are argv lists (no shell) that exit 0 only when the task is done.
Use stable, repeated step kinds for repeated kinds of work."""


def topological_order(tasks: Sequence[Task]) -> list[Task]:
    """Return tasks in dependency order, ties broken by id. Raises PlanError on a cycle."""
    by_id = {task.id: task for task in tasks}
    indegree = {task.id: 0 for task in tasks}
    dependents: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        for dependency in task.depends_on:
            if dependency not in by_id:
                raise PlanError(f"task {task.id!r} depends on unknown task {dependency!r}")
            indegree[task.id] += 1
            dependents[dependency].append(task.id)
    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[Task] = []
    while ready:
        task_id = heapq.heappop(ready)
        ordered.append(by_id[task_id])
        for child in dependents[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != len(tasks):
        cyclic = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
        raise PlanError(f"plan has a dependency cycle among: {', '.join(cyclic)}")
    return ordered


def validate_plan(plan: Plan) -> Plan:
    """Check ids are unique and the dependency graph is a DAG; return the plan unchanged."""
    seen: set[str] = set()
    for task in plan.tasks:
        if task.id in seen:
            raise PlanError(f"duplicate task id {task.id!r}")
        seen.add(task.id)
        if task.id in task.depends_on:
            raise PlanError(f"task {task.id!r} depends on itself")
    topological_order(plan.tasks)
    return plan


def plan_from_data(goal: str, data: object, redactor: Redactor) -> Plan:
    """Build a validated, redacted plan from model-produced JSON data."""
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise PlanError("plan JSON must be an object with a 'tasks' list")
    try:
        tasks = [Task.model_validate(item) for item in data["tasks"]]
        fresh = [
            task.model_copy(update={"state": TaskState.PENDING, "attempts": 0}) for task in tasks
        ]
        plan = Plan(goal=goal, tasks=fresh)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise PlanError(f"invalid plan: {details}") from None
    clean = Plan.model_validate(redactor.redact_obj(plan.model_dump(mode="json")))
    return validate_plan(clean)


def build_plan_messages(goal: str) -> list[Message]:
    """Prompt for the planner."""
    return [
        Message(role="system", content=PLANNER_SYSTEM_PROMPT),
        Message(
            role="user", content=f"Goal: {goal}\n{encode_request({'type': 'plan', 'goal': goal})}"
        ),
    ]


class Planner:
    """Asks a model for a plan and validates it."""

    def __init__(
        self,
        model: ModelClient,
        costs: CostTable,
        redactor: Redactor,
        *,
        tier: str = "large",
        max_tokens: int = 1024,
        budget: BudgetGovernor | None = None,
    ) -> None:
        """Configure the planner."""
        self._model = model
        self._costs = costs
        self._redactor = redactor
        self._tier = tier
        self._max_tokens = max_tokens
        self._budget = budget

    def plan(self, goal: str) -> tuple[Plan, Usage]:
        """Return a validated plan for ``goal`` and the usage of the planning call."""
        if not goal.strip():
            raise PlanError("goal must not be empty")
        if self._budget is not None:
            self._budget.check()
        completion = self._model.complete(build_plan_messages(goal), self._tier, self._max_tokens)
        usage = Usage(
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            cost=self._costs.cost(self._tier, completion.tokens_in, completion.tokens_out),
            model_calls=1,
        )
        if self._budget is not None:
            self._budget.charge(usage)
        try:
            data = extract_json_object(completion.text)
        except ValueError:
            raise PlanError("planner reply contained no JSON object") from None
        return plan_from_data(goal, data, self._redactor), usage


def replace_task(plan: Plan, task: Task) -> Plan:
    """Return ``plan`` with ``task`` substituted for the task with the same id."""
    return plan.model_copy(update={"tasks": [task if t.id == task.id else t for t in plan.tasks]})


class TaskLedger:
    """Validates task state transitions and persists each one to ``task_events``."""

    def __init__(self, db: Database, clock: Clock) -> None:
        """Bind to ``db``."""
        self._db = db
        self._clock = clock

    def transition(
        self, plan: Plan, task_id: str, target: TaskState, *, run_id: str | None, reason: str
    ) -> Plan:
        """Move ``task_id`` to ``target`` and return the updated plan."""
        task = plan.task(task_id)
        if target not in ALLOWED_TRANSITIONS[task.state]:
            raise PlanError(f"illegal transition for {task_id}: {task.state} -> {target}")
        self._db.execute(
            "INSERT INTO task_events (run_id, task_id, from_state, to_state, reason, ts)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, task_id, task.state.value, target.value, reason, iso(self._clock.now())),
        )
        return replace_task(plan, task.model_copy(update={"state": target}))

    def events(self, task_id: str | None = None) -> list[tuple[str, str, str, str]]:
        """Return ``(task_id, from, to, reason)`` tuples in order."""
        sql = "SELECT task_id, from_state, to_state, reason FROM task_events"
        params: tuple[str, ...] = ()
        if task_id is not None:
            sql += " WHERE task_id = ?"
            params = (task_id,)
        rows = self._db.query(sql + " ORDER BY id", params)
        return [(r["task_id"], r["from_state"], r["to_state"], r["reason"]) for r in rows]
