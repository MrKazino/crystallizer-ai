"""Trace recording/loading and the human approval tier."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from crystallizer.approval import DenyApprover, ScriptedApprover, TTYApprover, describe
from crystallizer.config import PolicyConfig
from crystallizer.errors import ExitCode, HumanApprovalError
from crystallizer.policy import Policy
from crystallizer.redaction import REDACTED, Redactor
from crystallizer.schemas import (
    Action,
    ApprovalRequest,
    Situation,
    StepResult,
    TraceOverhead,
    TraceStep,
    TraceVerdict,
    Usage,
)
from crystallizer.traces import TraceRecorder, TraceStore, normalize


def step(
    run: str, task: str, attempt: int, index: int, action: Action, **extra: object
) -> TraceStep:
    return TraceStep.model_validate(
        {
            "id": f"{run}:{task}:{attempt}:{index}",
            "run_id": run,
            "task_id": task,
            "attempt": attempt,
            "index": index,
            "ts": "t",
            "situation": Situation(task_kind="k", step_kind="s", step_index=index),
            "action": action,
            "result": StepResult(ok=True, output_hash="h"),
            "route": "small",
            "tokens_in": 10,
            "tokens_out": 5,
            "cost": 0.5,
            "model_calls": 2,
            **extra,
        }
    )


def test_recorder_redacts_and_flags(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "traces", "run-1", Redactor())
    clean = recorder.record_step(
        step("run-1", "t", 1, 0, Action(tool="file_read", args={"path": "a"}))
    )
    assert not clean.redacted
    secret = Action(tool="file_write", args={"path": "a", "content": "api_key=zzz"})
    stored = recorder.record_step(step("run-1", "t", 1, 1, secret))
    assert stored.redacted
    assert stored.action.args["content"] == f"api_key={REDACTED}"
    raw = recorder.path.read_text(encoding="utf-8")
    assert "zzz" not in raw


def test_store_derives_verified_and_usage(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    recorder = TraceRecorder(directory, "run-1", Redactor())
    read = Action(tool="file_read", args={"path": "a"})
    recorder.record_step(step("run-1", "t", 1, 0, read))
    recorder.record_verdict(
        TraceVerdict(run_id="run-1", task_id="t", attempt=1, passed=False, ts="t")
    )
    overhead = TraceOverhead(
        run_id="run-1",
        task_id="t",
        attempt=1,
        index=1,
        ts="t",
        reason="no proposal",
        usage=Usage(cost=1.0, model_calls=1),
    )
    assert recorder.record_overhead(overhead) == overhead
    empty = overhead.model_copy(update={"usage": Usage()})
    recorder.record_overhead(empty)  # nothing spent: nothing written
    recorder.record_step(step("run-1", "t", 2, 0, read))
    recorder.record_verdict(
        TraceVerdict(run_id="run-1", task_id="t", attempt=2, passed=True, ts="t")
    )
    with recorder.path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "step", "id": "torn')
    store = TraceStore(directory)
    assert store.run_ids() == ["run-1"]
    steps = store.load_run("run-1")
    assert [(s.attempt, s.verified) for s in steps] == [(1, False), (2, True)]
    usage = store.run_usage("run-1")
    assert usage.cost == pytest.approx(2.0)
    assert usage.model_calls == 5
    assert len(store.load_all()) == 2
    assert len(store.overheads()) == 1
    assert store.load_run("missing") == []
    assert TraceStore(tmp_path / "none").run_ids() == []


def test_normalize() -> None:
    view = normalize(
        step("r", "t", 1, 0, Action(tool="file_write", args={"path": "./a//b", "content": ""}))
    )
    assert view["action"]["args"]["path"] == "a/b"
    assert view["ok"] is True


def request() -> ApprovalRequest:
    action = Action(tool="file_delete", args={"path": "a"})
    return ApprovalRequest(
        run_id="r",
        task_id="t",
        step_index=0,
        situation=Situation(task_kind="k", step_kind="s"),
        actions=[action],
        decisions=[Policy(PolicyConfig()).classify(action)],
        reason="irreversible action",
    )


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_tty_approver_requires_tty() -> None:
    approver = TTYApprover(stdin=io.StringIO("y\n"), stdout=io.StringIO())
    with pytest.raises(HumanApprovalError) as info:
        approver.approve(request())
    assert info.value.exit_code is ExitCode.HUMAN_UNAVAILABLE
    with pytest.raises(HumanApprovalError):
        approver.propose(request())


@pytest.mark.parametrize(
    ("answer", "expected"), [("y\n", True), ("YES\n", True), ("\n", False), ("n\n", False)]
)
def test_tty_approver_answers(answer: str, expected: bool) -> None:
    out = io.StringIO()
    assert TTYApprover(stdin=FakeTTY(answer), stdout=out).approve(request()) is expected
    assert "file_delete" in out.getvalue()


def test_tty_approver_propose() -> None:
    good = TTYApprover(
        stdin=FakeTTY('{"tool": "file_read", "args": {"path": "a"}}\n'), stdout=io.StringIO()
    )
    assert good.propose(request()) == Action(tool="file_read", args={"path": "a"})
    assert TTYApprover(stdin=FakeTTY("\n"), stdout=io.StringIO()).propose(request()) is None
    assert TTYApprover(stdin=FakeTTY("{nope\n"), stdout=io.StringIO()).propose(request()) is None


def test_deny_and_scripted_approvers() -> None:
    with pytest.raises(HumanApprovalError):
        DenyApprover().approve(request())
    with pytest.raises(HumanApprovalError):
        DenyApprover().propose(request())
    scripted = ScriptedApprover([True], [Action(tool="run_tests")])
    assert scripted.approve(request())
    assert not scripted.approve(request())
    assert scripted.propose(request()) == Action(tool="run_tests")
    assert scripted.propose(request()) is None
    assert len(scripted.requests) == 4
    assert "irreversible action" in describe(request())
