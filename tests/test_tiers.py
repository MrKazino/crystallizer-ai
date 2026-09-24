"""Shared tier helpers: prompts, proposal parsing, voting keys."""

from __future__ import annotations

from crystallizer.models import decode_request
from crystallizer.schemas import Action, Situation, StepRequest, Task
from crystallizer.tiers import build_action_messages, parse_proposal, proposal_key


def request() -> StepRequest:
    task = Task(id="t1", title="T", kind="k", params={"name": "x"})
    return StepRequest(
        run_id="r",
        task=task,
        situation=Situation(task_kind="k", step_kind="open", params={"name": "x"}),
        attempt=2,
        context="CONTEXT",
        open_mode=True,
        tools=["file_read"],
    )


def test_action_messages_carry_request_line() -> None:
    messages = build_action_messages(request(), "small")
    assert "file_read" in messages[0].content
    assert "(open mode)" in messages[1].content
    decoded = decode_request(messages)
    assert decoded is not None
    assert decoded["attempt"] == 2
    assert decoded["task_id"] == "t1"


def test_parse_proposal() -> None:
    assert parse_proposal('{"done": true}') == ([], True)
    assert parse_proposal("no json") is None
    assert parse_proposal('{"done": true, "extra": 1}') is None
    assert parse_proposal('{"tool": "x", "args": {}, "note": "hi"}') is None
    assert parse_proposal('{"tool": "", "args": {}}') is None
    parsed = parse_proposal('Sure: {"tool": "file_read", "args": {"path": "./a"}}')
    assert parsed == ([Action(tool="file_read", args={"path": "a"})], False)


def test_proposal_keys() -> None:
    a = Action(tool="file_read", args={"path": "./a"})
    b = Action(tool="file_read", args={"path": "a"})
    assert proposal_key([a], False) == proposal_key([b], False)
    assert proposal_key([], True) != proposal_key([a], False)
