"""Runner and Harness: scripted projects, crash-and-resume at every fault point, gates, budgets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from crystallizer.approval import ScriptedApprover
from crystallizer.errors import (
    BudgetExhaustedError,
    HumanApprovalError,
    SimulatedCrash,
    UsageError,
    WorkspaceLockedError,
)
from crystallizer.events import EventKind
from crystallizer.faults import FAULT_POINTS, FaultInjector
from crystallizer.lock import RunLock
from crystallizer.models import MockBehavior, MockModel, MockScript, MockStep
from crystallizer.schemas import Action, Event, TaskState, ToolResult
from crystallizer.tools import ToolRegistry
from tests.helpers import (
    GOAL,
    act,
    check_file,
    fast_workspace,
    greeting_steps,
    open_harness,
    project_script,
    task,
)


@pytest.fixture
def executions(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count real tool executions (journal replays do not call tools)."""
    counts: Counter[str] = Counter()
    original = ToolRegistry.execute

    def counting(self: ToolRegistry, action: Action) -> ToolResult:
        if action.tool != "shell":
            counts[json.dumps(action.model_dump(mode="json"), sort_keys=True)] += 1
        return original(self, action)

    monkeypatch.setattr(ToolRegistry, "execute", counting)
    return counts


def plan_project(workspace: Path, script: MockScript | None = None) -> None:
    with open_harness(workspace, script or project_script()) as harness:
        report = harness.plan(GOAL)
        assert report.plan is not None


def test_scripted_project_runs_to_completion(workspace: Path) -> None:
    fast_workspace(workspace)
    script = project_script()
    plan_project(workspace, script)
    events: list[Event] = []
    with open_harness(workspace, script) as harness:
        harness.bus.subscribe(events.append)
        summary = harness.run()
        status = harness.status()
    assert summary.exit_code == 0
    assert [t.state for t in summary.tasks] == [TaskState.DONE, TaskState.DONE]
    assert summary.route_mix == {"small": 4}
    assert summary.usage.model_calls == 12  # three small samples per step
    assert (workspace / "out/alpha.txt").read_text(encoding="utf-8") == "alpha"
    assert (workspace / "out/beta.txt").read_text(encoding="utf-8") == "beta"
    assert status.counts == {"done": 2}
    assert not status.run_active
    kinds = [event.kind for event in events]
    assert kinds[0] == EventKind.RUN_STARTED
    assert kinds[-1] == EventKind.RUN_FINISHED
    assert kinds.count(EventKind.STEP_EXECUTED) == 4


def test_run_again_is_a_no_op_new_run(workspace: Path) -> None:
    fast_workspace(workspace)
    plan_project(workspace)
    with open_harness(workspace, project_script()) as harness:
        harness.run()
        second = harness.run()
    assert second.steps == 0
    assert second.exit_code == 0


@pytest.mark.parametrize("point", sorted(FAULT_POINTS))
def test_crash_then_resume_has_no_duplicate_side_effects(
    workspace: Path, executions: Counter[str], point: str
) -> None:
    fast_workspace(workspace)
    script = project_script()
    plan_project(workspace, script)
    with open_harness(workspace, script, faults=FaultInjector(point, after=2)) as harness:
        with pytest.raises(SimulatedCrash):
            harness.run()
    assert not (workspace / ".crystallizer" / "run.lock").exists()
    with open_harness(workspace, script) as harness:
        with pytest.raises(UsageError, match="unfinished"):
            harness.run()
        summary = harness.resume()
        assert harness.resume().nothing_to_do
    assert summary.exit_code == 0
    assert summary.resumed
    assert [t.state for t in summary.tasks] == [TaskState.DONE, TaskState.DONE]
    duplicates = {action: count for action, count in executions.items() if count > 1}
    if point == "journal.after_execute":
        # The crashed action ran but was never recorded as completed: it is in doubt, and being
        # an idempotent file tool it is safely redone exactly once.
        assert len(duplicates) == 1
        assert set(duplicates.values()) == {2}
    else:
        assert duplicates == {}
    assert (workspace / "out/beta.txt").read_text(encoding="utf-8") == "beta"


def test_in_doubt_non_idempotent_action_needs_a_human(workspace: Path) -> None:
    fast_workspace(workspace, '\n[tools]\nallowed_executables = ["python", "git"]\n')
    commit = act("shell", argv=["python", "-c", "print('side effect')"])
    script = project_script({"t1": [MockStep(action=commit), *greeting_steps("alpha")[:1]]})
    plan_project(workspace, script)
    with open_harness(workspace, script, faults=FaultInjector("journal.after_execute")) as harness:
        with pytest.raises(SimulatedCrash):
            harness.run()
    with open_harness(workspace, script) as harness:
        with pytest.raises(HumanApprovalError):
            harness.resume()
    approver = ScriptedApprover(answers=[True])
    with open_harness(workspace, script, approver=approver) as harness:
        summary = harness.resume()
    assert summary.exit_code == 0
    assert "started before a crash" in approver.requests[0].reason


def test_failed_task_blocks_dependents_exit_3(workspace: Path) -> None:
    fast_workspace(workspace)
    wrong = greeting_steps("alpha", small=MockBehavior.WRONG, large=MockBehavior.WRONG)
    script = project_script({"t1": wrong})
    plan_project(workspace, script)
    with open_harness(workspace, script) as harness:
        summary = harness.run()
    assert summary.exit_code == 3
    assert summary.failed == ["t1"]
    assert summary.blocked == ["t2"]
    t1 = next(t for t in summary.tasks if t.id == "t1")
    assert t1.attempts == 6  # floors skill, small, large; one retry at each
    with open_harness(workspace, project_script()) as harness:
        rerun = harness.run()
    assert rerun.exit_code == 0
    assert [t.state for t in rerun.tasks] == [TaskState.DONE, TaskState.DONE]


def test_acceptance_failure_fails_task(workspace: Path) -> None:
    fast_workspace(workspace)
    wrong = [
        MockStep(action=act("file_write", path="out/alpha.txt", content="nope")),
        MockStep(action=act("file_read", path="out/alpha.txt")),
    ]
    script = project_script({"t1": wrong})
    plan_project(workspace, script)
    with open_harness(workspace, script) as harness:
        summary = harness.run()
    assert summary.failed == ["t1"]


def test_tool_failure_fails_attempt(workspace: Path) -> None:
    fast_workspace(workspace)
    script = project_script({"t1": [MockStep(action=act("file_read", path="missing.txt"))]})
    plan_project(workspace, script)
    with open_harness(workspace, script) as harness:
        assert harness.run().failed == ["t1"]


def test_irreversible_action_forced_to_human(workspace: Path) -> None:
    fast_workspace(workspace)
    (workspace / "old.txt").write_text("x", encoding="utf-8")
    steps = [MockStep(action=act("file_delete", path="old.txt")), *greeting_steps("alpha")[:1]]
    script = project_script({"t1": steps})
    plan_project(workspace, script)
    with open_harness(workspace, script) as harness:
        with pytest.raises(HumanApprovalError):
            harness.run()
    assert (workspace / "old.txt").exists()
    approver = ScriptedApprover(answers=[True])
    with open_harness(workspace, script, approver=approver) as harness:
        summary = harness.resume()
    assert summary.exit_code == 0
    assert not (workspace / "old.txt").exists()
    assert approver.requests[0].decisions[0].action_name == "file_delete"


def test_acceptance_commands_go_through_policy(workspace: Path) -> None:
    plan_project(workspace)
    with open_harness(workspace, project_script()) as harness:
        with pytest.raises(HumanApprovalError, match="acceptance"):
            harness.run()


def test_budget_exhaustion_exit_7_then_resume(workspace: Path) -> None:
    fast_workspace(workspace, "\n[budget]\nmax_model_calls_per_run = 3\n")
    plan_project(workspace)
    with open_harness(workspace, project_script()) as harness:
        with pytest.raises(BudgetExhaustedError):
            harness.run()
        assert harness.status().run_active
    fast_workspace(workspace, "\n[budget]\nmax_model_calls_per_run = 100\n")
    with open_harness(workspace, project_script()) as harness:
        summary = harness.resume()
    assert summary.exit_code == 0
    assert summary.usage.model_calls == 12  # nothing was re-asked on resume


def test_open_mode_task(workspace: Path) -> None:
    fast_workspace(workspace)
    open_task = task("t1", "alpha", steps=())
    steps = [*greeting_steps("alpha"), MockStep(done=True)]
    script = MockScript(plans={GOAL: json.dumps({"tasks": [open_task]})}, actions={"t1": steps})
    plan_project(workspace, script)
    with open_harness(workspace, script) as harness:
        summary = harness.run()
    assert summary.exit_code == 0
    assert summary.steps == 2


def test_done_at_planned_step_fails(workspace: Path) -> None:
    fast_workspace(workspace)
    script = project_script({"t1": [MockStep(done=True)]})
    plan_project(workspace, script)
    with open_harness(workspace, script) as harness:
        assert harness.run().failed == ["t1"]


def test_second_concurrent_run_is_locked(workspace: Path) -> None:
    fast_workspace(workspace)
    plan_project(workspace)
    with RunLock(workspace / ".crystallizer"), open_harness(workspace, project_script()) as harness:
        with pytest.raises(WorkspaceLockedError):
            harness.run()


def test_run_without_plan(workspace: Path) -> None:
    with open_harness(workspace, project_script()) as harness:
        with pytest.raises(UsageError, match="no plan"):
            harness.run()
        with pytest.raises(UsageError, match="no plan"):
            harness.preview()
        assert not harness.status().has_plan


def test_plan_refused_while_run_unfinished(workspace: Path) -> None:
    fast_workspace(workspace, "\n[budget]\nmax_model_calls_per_run = 1\n")
    plan_project(workspace)
    with open_harness(workspace, project_script()) as harness:
        with pytest.raises(BudgetExhaustedError):
            harness.run()
        with pytest.raises(UsageError, match="unfinished"):
            harness.plan(GOAL)


def test_dry_run_writes_nothing(workspace: Path) -> None:
    fast_workspace(workspace)
    plan_project(workspace)
    state = workspace / ".crystallizer"
    before = sorted(str(p) for p in state.rglob("*"))
    model = MockModel(project_script())
    with open_harness(workspace, project_script(), dry_run=True, model=model) as harness:
        preview = harness.preview()
        plan_report = harness.plan("another goal")
        init_report = harness.init()
    assert [line.detail for line in preview] == ["would call small model"] * 4
    assert plan_report.dry_run
    assert "would call large model" in plan_report.detail
    assert init_report.dry_run
    assert model.calls == []
    assert sorted(str(p) for p in state.rglob("*")) == before


def test_init_creates_layout(workspace: Path) -> None:
    with open_harness(workspace, project_script()) as harness:
        report = harness.init()
        again = harness.init()
    assert "crystallizer.toml" in report.created
    assert (workspace / ".crystallizer" / "skills" / "active").is_dir()
    assert again.created == []


def test_memory_records_tasks(workspace: Path) -> None:
    fast_workspace(workspace)
    plan_project(workspace)
    with open_harness(workspace, project_script()) as harness:
        harness.run()
        entries = harness._open_state().memory.entries()
    kinds = [entry.kind.value for entry in entries]
    assert kinds.count("task") == 2
    assert kinds.count("decision") == 1


def test_tool_specs_listed(workspace: Path) -> None:
    with open_harness(workspace, project_script()) as harness:
        assert "file_write" in [spec.name for spec in harness.tool_specs()]


def test_check_file_helper() -> None:
    assert check_file("a", "b")[0] == "python"


def test_adopt_plan(workspace: Path) -> None:
    from crystallizer.planner import plan_from_data
    from crystallizer.redaction import Redactor

    fast_workspace(workspace)
    external = plan_from_data("external goal", {"tasks": [task("t1", "alpha")]}, Redactor())
    done = external.model_copy(
        update={"tasks": [external.tasks[0].model_copy(update={"state": TaskState.DONE})]}
    )
    with open_harness(workspace, project_script(), dry_run=True) as harness:
        assert harness.adopt_plan(done).tasks[0].state is TaskState.PENDING
    with open_harness(workspace, project_script()) as harness:
        adopted = harness.adopt_plan(done)
        assert adopted.tasks[0].state is TaskState.PENDING
        summary = harness.run()
    assert summary.exit_code == 0
    fast_workspace(workspace, "\n[budget]\nmax_model_calls_per_run = 1\n")
    with open_harness(workspace, project_script()) as harness:
        harness.adopt_plan(external)
        with pytest.raises(BudgetExhaustedError):
            harness.run()
        with pytest.raises(UsageError, match="unfinished"):
            harness.adopt_plan(external)
