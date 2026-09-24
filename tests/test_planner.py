"""Planner: JSON extraction, validation, DAG checks, ordering, transitions."""

from __future__ import annotations

import json

import pytest

from crystallizer.budget import BudgetGovernor
from crystallizer.clock import FixedClock
from crystallizer.config import BudgetConfig, CostConfig
from crystallizer.db import Database
from crystallizer.errors import BudgetExhaustedError, PlanError
from crystallizer.models import CostTable, MockModel, MockScript, extract_json_object
from crystallizer.planner import (
    Planner,
    TaskLedger,
    plan_from_data,
    replace_task,
    topological_order,
    validate_plan,
)
from crystallizer.redaction import REDACTED, Redactor
from crystallizer.schemas import Plan, Task, TaskState
from tests.helpers import GOAL, project_script, task


def make(tasks: list[dict[str, object]]) -> Plan:
    return plan_from_data("g", {"tasks": tasks}, Redactor())


def test_extract_json_object() -> None:
    assert extract_json_object('noise {"a": 1} {"b": 2}') == {"a": 1}
    assert extract_json_object('{bad {"ok": true}') == {"ok": True}
    with pytest.raises(ValueError, match="no JSON"):
        extract_json_object("[1, 2] no object")


def test_planner_uses_model_and_reports_usage() -> None:
    planner = Planner(MockModel(project_script()), CostTable(CostConfig()), Redactor())
    plan, usage = planner.plan(GOAL)
    assert [t.id for t in plan.tasks] == ["t1", "t2"]
    assert all(t.state is TaskState.PENDING for t in plan.tasks)
    assert usage.model_calls == 1
    assert usage.cost == pytest.approx((400 * 3.0 + 300 * 15.0) / 1_000_000)


def test_planner_budget_and_errors() -> None:
    costs = CostTable(CostConfig())
    budget = BudgetGovernor(BudgetConfig(max_model_calls_per_run=1))
    planner = Planner(MockModel(project_script()), costs, Redactor(), budget=budget)
    planner.plan(GOAL)
    with pytest.raises(BudgetExhaustedError):
        planner.plan(GOAL)
    with pytest.raises(PlanError, match="empty"):
        planner.plan("  ")
    garbage = MockModel(MockScript(plans={"g": "no json here"}))
    with pytest.raises(PlanError, match="no JSON"):
        Planner(garbage, costs, Redactor()).plan("g")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"nope": []}, "'tasks' list"),
        ([], "'tasks' list"),
        ({"tasks": [{"id": "a"}]}, "invalid plan"),
        ({"tasks": []}, "invalid plan"),
    ],
)
def test_invalid_plan_data(data: object, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        plan_from_data("g", data, Redactor())


def test_dag_validation() -> None:
    with pytest.raises(PlanError, match="duplicate"):
        make([task("a", "x"), task("a", "y")])
    with pytest.raises(PlanError, match="unknown task"):
        make([task("a", "x", depends_on=["zzz"])])
    with pytest.raises(PlanError, match="itself"):
        make([task("a", "x", depends_on=["a"])])
    with pytest.raises(PlanError, match="cycle"):
        make([task("a", "x", depends_on=["b"]), task("b", "y", depends_on=["a"])])


def test_topological_order_is_deterministic() -> None:
    plan = make(
        [
            task("d", "d", depends_on=["b", "c"]),
            task("c", "c", depends_on=["a"]),
            task("b", "b", depends_on=["a"]),
            task("a", "a"),
            task("e", "e"),
        ]
    )
    assert [t.id for t in topological_order(plan.tasks)] == ["a", "b", "c", "d", "e"]
    assert validate_plan(plan) is plan


def test_plan_is_redacted_and_state_reset() -> None:
    data = task("a", "x")
    data["title"] = "Use token=abc123 here"
    data["state"] = "done"
    plan = make([data])
    assert "abc123" not in plan.tasks[0].title
    assert REDACTED in plan.tasks[0].title
    assert plan.tasks[0].state is TaskState.PENDING


def test_ledger_transitions(db: Database) -> None:
    ledger = TaskLedger(db, FixedClock())
    plan = make([task("a", "x")])
    plan = ledger.transition(plan, "a", TaskState.RUNNING, run_id="r", reason="start")
    plan = ledger.transition(plan, "a", TaskState.FAILED, run_id="r", reason="bad")
    plan = ledger.transition(plan, "a", TaskState.PENDING, run_id="r2", reason="new run")
    with pytest.raises(PlanError, match="illegal"):
        ledger.transition(plan, "a", TaskState.DONE, run_id="r2", reason="skip")
    assert ledger.events("a") == [
        ("a", "pending", "running", "start"),
        ("a", "running", "failed", "bad"),
        ("a", "failed", "pending", "new run"),
    ]
    assert len(ledger.events()) == 3


def test_replace_task() -> None:
    plan = make([task("a", "x"), task("b", "y")])
    updated = replace_task(plan, plan.task("b").model_copy(update={"attempts": 4}))
    assert updated.task("b").attempts == 4
    assert plan.task("b").attempts == 0
    assert isinstance(json.dumps(updated.model_dump(mode="json")), str)
    assert isinstance(Task.model_validate(updated.task("a").model_dump()), Task)
