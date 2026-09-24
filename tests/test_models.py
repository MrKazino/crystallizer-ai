"""MockModel behaviors, request protocol, cost accounting, budget governor."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from crystallizer.budget import BudgetGovernor
from crystallizer.config import BudgetConfig, CostConfig, ModelConfig
from crystallizer.errors import BudgetExhaustedError, ExitCode, ModelError
from crystallizer.models import (
    CostTable,
    MockBehavior,
    MockModel,
    MockScript,
    MockStep,
    decode_request,
    encode_request,
    model_identifier,
)
from crystallizer.schemas import Action, Message, Usage

WRITE = Action(tool="file_write", args={"path": "a.py", "content": "x"})


def ask(model: MockModel, tier: str, **payload: object) -> str:
    messages = [Message(role="user", content="context\n" + encode_request(dict(payload)))]
    return model.complete(messages, tier, 1024).text


def script(**step: object) -> MockScript:
    return MockScript(actions={"t1": [MockStep.model_validate({"action": WRITE, **step})]})


def test_request_roundtrip() -> None:
    messages = [Message(role="user", content="x\n" + encode_request({"type": "plan"}))]
    assert decode_request(messages) == {"type": "plan"}
    assert decode_request([Message(role="user", content="nothing")]) is None
    bad = [Message(role="user", content="CRYSTALLIZER-REQUEST: {not json")]
    assert decode_request(bad) is None
    assert decode_request([Message(role="user", content="CRYSTALLIZER-REQUEST: [1]")]) is None


def test_correct_answers_agree() -> None:
    model = MockModel(script())
    answers = {ask(model, "small", type="action", task_id="t1", step_index=0) for _ in range(3)}
    assert len(answers) == 1
    assert json.loads(answers.pop()) == {"tool": "file_write", "args": WRITE.args}


def test_disagreement_yields_distinct_answers() -> None:
    model = MockModel(script(small=MockBehavior.DISAGREE), seed=3)
    answers = [ask(model, "small", type="action", task_id="t1", step_index=0) for _ in range(3)]
    assert len(set(answers)) == 3
    assert json.loads(answers[0])["args"] == WRITE.args
    large = ask(model, "large", type="action", task_id="t1", step_index=0)
    assert json.loads(large)["args"] == WRITE.args


def test_wrong_answers_agree_on_wrong_action() -> None:
    model = MockModel(script(small=MockBehavior.WRONG))
    answers = {ask(model, "small", type="action", task_id="t1", step_index=0) for _ in range(3)}
    assert len(answers) == 1
    assert json.loads(answers.pop())["args"] != WRITE.args


def test_variants_for_list_and_argless_actions() -> None:
    shell = Action(tool="shell", args={"argv": ["git", "status"]})
    model = MockModel(
        MockScript(
            actions={
                "t1": [MockStep(action=shell, small=MockBehavior.WRONG)],
                "t2": [MockStep(action=Action(tool="run_tests"), small=MockBehavior.WRONG)],
            }
        )
    )
    assert json.loads(ask(model, "small", type="action", task_id="t1"))["args"]["argv"][0] == "git"
    assert "variant" in json.loads(ask(model, "small", type="action", task_id="t2"))["args"]


def test_error_garbage_done_and_end_of_script() -> None:
    model = MockModel(
        MockScript(
            actions={
                "t1": [
                    MockStep(action=WRITE, small=MockBehavior.ERROR),
                    MockStep(action=WRITE, small=MockBehavior.GARBAGE),
                    MockStep(done=True),
                ]
            }
        )
    )
    with pytest.raises(ModelError):
        ask(model, "small", type="action", task_id="t1", step_index=0)
    with pytest.raises(json.JSONDecodeError):
        json.loads(ask(model, "small", type="action", task_id="t1", step_index=1))
    assert json.loads(ask(model, "small", type="action", task_id="t1", step_index=2)) == {
        "done": True
    }
    assert json.loads(ask(model, "small", type="action", task_id="t1", step_index=9)) == {
        "done": True
    }


def test_plan_and_summarize_and_errors() -> None:
    model = MockModel(MockScript(plans={"g": '{"tasks": []}'}))
    assert ask(model, "large", type="plan", goal="g") == '{"tasks": []}'
    assert "(2 entries)" in ask(model, "small", type="summarize", entries=["a", "b"])
    with pytest.raises(ModelError):
        ask(model, "large", type="plan", goal="other")
    with pytest.raises(ModelError):
        ask(model, "small", type="action", task_id="missing")
    with pytest.raises(ModelError):
        ask(model, "small", type="dance")
    with pytest.raises(ModelError):
        ask(model, "medium", type="plan", goal="g")
    with pytest.raises(ModelError):
        model.complete([Message(role="user", content="no request")], "small", 10)
    assert ("plan", "large") in model.calls


def test_max_tokens_caps_output() -> None:
    model = MockModel(script(tokens_out=500))
    messages = [Message(role="user", content=encode_request({"type": "action", "task_id": "t1"}))]
    assert model.complete(messages, "small", 100).tokens_out == 100


def test_mock_step_needs_one_answer() -> None:
    with pytest.raises(ValidationError):
        MockStep()
    with pytest.raises(ValidationError):
        MockStep(action=WRITE, done=True)


def test_cost_table() -> None:
    table = CostTable(CostConfig())
    assert table.cost("small", 1_000_000, 0) == pytest.approx(0.25)
    assert table.cost("large", 0, 1_000_000) == pytest.approx(15.0)
    with pytest.raises(ModelError):
        table.cost("medium", 1, 1)


def test_model_identifier() -> None:
    config = ModelConfig(small="s-model", large="l-model")
    assert model_identifier(config, "small") == "s-model"
    assert model_identifier(config, "large") == "l-model"
    with pytest.raises(ModelError):
        model_identifier(config, "x")


@pytest.mark.parametrize(
    ("limits", "usage"),
    [
        (BudgetConfig(max_cost_per_run=1.0), Usage(cost=1.0)),
        (BudgetConfig(max_tokens_per_run=10), Usage(tokens_in=6, tokens_out=4)),
        (BudgetConfig(max_model_calls_per_run=2), Usage(model_calls=2)),
    ],
)
def test_budget_limits(limits: BudgetConfig, usage: Usage) -> None:
    governor = BudgetGovernor(limits)
    governor.check()
    governor.charge(usage)
    with pytest.raises(BudgetExhaustedError) as info:
        governor.check()
    assert info.value.exit_code is ExitCode.BUDGET_EXHAUSTED


def test_budget_unlimited_by_default_and_resumable() -> None:
    governor = BudgetGovernor(BudgetConfig(), spent=Usage(cost=1e9, model_calls=10**6))
    governor.check()
    assert governor.spent.model_calls == 10**6
