"""Data model validation, schema export and drift checking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crystallizer.schemas import (
    Action,
    Condition,
    GuardOp,
    Plan,
    Proposal,
    RangeValue,
    Situation,
    Skill,
    SkillManifest,
    Task,
    TierResult,
    Usage,
    check_schemas,
    export_schemas,
    exported_models,
    render_schema,
)


def test_situation_lookup_and_fields() -> None:
    situation = Situation(task_kind="t", step_kind="s", step_index=2, params={"name": "x", "n": 3})
    assert situation.lookup("task_kind") == (True, "t")
    assert situation.lookup("step_kind") == (True, "s")
    assert situation.lookup("step_index") == (True, 2)
    assert situation.lookup("params.name") == (True, "x")
    assert situation.lookup("params.missing") == (False, None)
    assert situation.lookup("other") == (False, None)
    assert situation.fields() == {
        "task_kind": "t",
        "step_kind": "s",
        "step_index": 2,
        "params.name": "x",
        "params.n": 3,
    }


def test_scalar_types_are_preserved() -> None:
    situation = Situation(task_kind="t", step_kind="s", params={"b": True, "i": 1, "f": 1.5})
    assert situation.params["b"] is True
    assert type(situation.params["i"]) is int
    assert type(situation.params["f"]) is float


def test_task_validation() -> None:
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", kind="k", params={"bad-name": 1})
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", kind="k", acceptance_commands=[[]])
    with pytest.raises(ValidationError):
        Task(id="../x", title="x", kind="k")
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", kind="k", extra_field=1)  # type: ignore[call-arg]


def test_plan_task_lookup() -> None:
    plan = Plan(goal="g", tasks=[Task(id="a", title="A", kind="k")])
    assert plan.task("a").title == "A"
    with pytest.raises(KeyError):
        plan.task("b")


@pytest.mark.parametrize(
    ("op", "value"),
    [
        (GuardOp.EQ, [1]),
        (GuardOp.IN, []),
        (GuardOp.IN, "x"),
        (GuardOp.RANGE, 3),
        (GuardOp.HAS_TYPE, "float"),
        (GuardOp.EXISTS, False),
    ],
)
def test_condition_value_must_match_op(op: GuardOp, value: object) -> None:
    with pytest.raises(ValidationError):
        Condition.model_validate({"field": "task_kind", "op": op.value, "value": value})


def test_condition_accepts_valid_values() -> None:
    Condition(field="params.n", op=GuardOp.RANGE, value=RangeValue(min=1, max=2))
    Condition(field="params.n", op=GuardOp.EXISTS)
    with pytest.raises(ValidationError):
        RangeValue(min=3, max=1)
    with pytest.raises(ValidationError):
        Condition.model_validate({"field": "params.x-y", "op": "exists"})


def test_skill_and_manifest_keys() -> None:
    skill = Skill.model_validate(
        {
            "id": "skill-abc",
            "version": 2,
            "task_kind": "k",
            "guard": {"all": [{"field": "task_kind", "op": "eq", "value": "k"}]},
            "steps": [{"tool": "file_read", "step_kind": "s", "args": {"path": "x"}}],
        }
    )
    assert skill.key == "skill-abc@v2"
    manifest = SkillManifest(
        id="skill-abc",
        version=2,
        content_hash="h",
        guard_summary="g",
        source_file="f",
        created_at="t",
    )
    assert manifest.key == "skill-abc@v2"


def test_proposal_and_tier_result_shapes() -> None:
    Proposal(tier="small", actions=[Action(tool="t")], confidence=1.0)
    Proposal(tier="small", done=True, confidence=1.0)
    with pytest.raises(ValidationError):
        Proposal(tier="small", confidence=1.0)
    with pytest.raises(ValidationError):
        Proposal(tier="small", actions=[Action(tool="t")], done=True, confidence=1.0)
    with pytest.raises(ValidationError):
        TierResult(tier="small")


def test_usage_arithmetic() -> None:
    total = Usage(tokens_in=1, tokens_out=2, cost=0.5, model_calls=1).plus(
        Usage(tokens_in=3, tokens_out=4, cost=0.25, model_calls=2)
    )
    assert total == Usage(tokens_in=4, tokens_out=6, cost=0.75, model_calls=3)
    assert total.tokens == 10


def test_action_is_frozen() -> None:
    action = Action(tool="t", args={"a": 1})
    with pytest.raises(ValidationError):
        action.tool = "u"  # type: ignore[misc]


def test_export_and_check_roundtrip(tmp_path: Path) -> None:
    directory = tmp_path / "schemas"
    assert check_schemas(directory)
    written = export_schemas(directory)
    assert len(written) == len(exported_models())
    assert check_schemas(directory) == []
    (directory / "plan.schema.json").write_text("{}", encoding="utf-8")
    (directory / "extra.schema.json").write_text("{}", encoding="utf-8")
    (directory / "skill.schema.json").unlink()
    problems = check_schemas(directory)
    assert "differs: plan.schema.json" in problems
    assert "unexpected: extra.schema.json" in problems
    assert "missing: skill.schema.json" in problems


def test_committed_schemas_are_current() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    assert check_schemas(root) == []


def test_render_schema_is_deterministic() -> None:
    assert render_schema(Plan) == render_schema(Plan)
