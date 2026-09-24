"""Executor: validation (skill safety), guard semantics, instantiation, and properties."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crystallizer.errors import SkillError
from crystallizer.schemas import Action, Condition, Guard, GuardOp, RangeValue, Situation, SlotType
from crystallizer.skills.executor import (
    check_type,
    escape_literal,
    evaluate_guard,
    instantiate,
    parse_skill,
    parse_template,
    render_template,
    to_snake,
)


def skill_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "skill-abc123def456",
        "version": 1,
        "task_kind": "add_module",
        "slots": {"name": {"type": "identifier", "source": "params.name"}},
        "guard": {
            "all": [
                {"field": "task_kind", "op": "eq", "value": "add_module"},
                {"field": "params.name", "op": "has_type", "value": "identifier"},
            ]
        },
        "steps": [
            {
                "tool": "file_write",
                "step_kind": "scaffold",
                "args": {"path": "src/{name}.py", "content": "class {name|upper}: {{}}"},
            },
            {
                "tool": "shell",
                "step_kind": "commit",
                "args": {"argv": ["git", "commit", "-m", "add {name|snake}"]},
            },
        ],
    }
    data.update(overrides)
    return data


def situation(**params: Any) -> Situation:
    return Situation(task_kind="add_module", step_kind="scaffold", params=params)


def test_instantiate_renders_filters_and_escapes() -> None:
    skill = parse_skill(json.dumps(skill_data()))
    actions = instantiate(skill, situation(name="Parser"))
    assert actions == [
        Action(tool="file_write", args={"path": "src/Parser.py", "content": "class PARSER: {}"}),
        Action(tool="shell", args={"argv": ["git", "commit", "-m", "add parser"]}),
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["guard"]["all"].append({"field": "task_kind", "op": "regex", "value": "x"}),
            "invalid skill",
        ),
        (lambda d: d["steps"][0]["args"].update(path="{name|title}"), "unknown filter"),
        (lambda d: d["steps"][0]["args"].update(path="{other}.py"), "unknown slot"),
        (lambda d: d["steps"][0]["args"].update(path="{name"), "unterminated"),
        (lambda d: d["steps"][0]["args"].update(path="name}"), "unmatched"),
        (lambda d: d["steps"][0]["args"].update(path="{na me}"), "malformed"),
        (
            lambda d: (
                d["slots"].update(n={"type": "int", "source": "params.n"})
                or d["steps"][0]["args"].update(path="{n|upper}")
            ),
            "int slot",
        ),
        (lambda d: d["slots"].update(n={"type": "float", "source": "params.n"}), "invalid skill"),
        (
            lambda d: d["slots"].update(n={"type": "int", "source": "__import__('os')"}),
            "invalid skill",
        ),
        (lambda d: d.update(extra="code"), "invalid skill"),
    ],
)
def test_unsafe_or_malformed_skills_are_rejected(mutate: Any, message: str) -> None:
    data = skill_data()
    mutate(data)
    with pytest.raises(SkillError, match=message):
        parse_skill(data)


def test_parse_skill_rejects_bad_json() -> None:
    with pytest.raises(SkillError):
        parse_skill("{not json")


def test_guard_false_and_bad_slots() -> None:
    skill = parse_skill(skill_data())
    with pytest.raises(SkillError, match="guard"):
        instantiate(skill, Situation(task_kind="other", step_kind="scaffold", params={"name": "x"}))
    with pytest.raises(SkillError, match="guard"):
        instantiate(skill, situation(name="not an identifier"))
    loose = parse_skill(
        skill_data(guard={"all": [{"field": "task_kind", "op": "eq", "value": "add_module"}]})
    )
    with pytest.raises(SkillError, match=r"no params\.name"):
        instantiate(loose, situation())
    with pytest.raises(SkillError, match="type identifier"):
        instantiate(loose, situation(name="has space"))


def cond(field: str, op: GuardOp, value: Any = None) -> Condition:
    return Condition.model_validate({"field": field, "op": op.value, "value": value})


@pytest.mark.parametrize(
    ("condition", "params", "expected"),
    [
        (cond("params.flag", GuardOp.EQ, True), {"flag": True}, True),
        (cond("params.flag", GuardOp.EQ, True), {"flag": 1}, False),
        (cond("params.n", GuardOp.EQ, 1), {"n": True}, False),
        (cond("params.n", GuardOp.EQ, 1), {"n": 1.0}, True),
        (cond("params.s", GuardOp.EQ, "a"), {"s": "a"}, True),
        (cond("params.s", GuardOp.IN, ["a", "b"]), {"s": "b"}, True),
        (cond("params.s", GuardOp.IN, ["a", "b"]), {"s": "c"}, False),
        (cond("params.n", GuardOp.RANGE, {"min": 1, "max": 3}), {"n": 3}, True),
        (cond("params.n", GuardOp.RANGE, {"min": 1, "max": 3}), {"n": 4}, False),
        (cond("params.n", GuardOp.RANGE, {"min": 0, "max": 3}), {"n": True}, False),
        (cond("params.n", GuardOp.RANGE, {"min": 0, "max": 3}), {"n": "2"}, False),
        (cond("params.n", GuardOp.HAS_TYPE, "int"), {"n": 2}, True),
        (cond("params.n", GuardOp.EXISTS), {"n": 2}, True),
        (cond("params.n", GuardOp.EXISTS), {}, False),
        (cond("params.s", GuardOp.EQ, "a"), {}, False),
    ],
)
def test_condition_semantics(condition: Condition, params: dict[str, Any], expected: bool) -> None:
    assert evaluate_guard(Guard(all=[condition]), situation(**params)) is expected


def test_step_fields_in_guards() -> None:
    guard = Guard(
        all=[
            cond("step_kind", GuardOp.EQ, "scaffold"),
            cond("step_index", GuardOp.RANGE, RangeValue(min=0, max=0).model_dump()),
        ]
    )
    assert evaluate_guard(guard, situation())
    assert not evaluate_guard(guard, Situation(task_kind="k", step_kind="scaffold", step_index=1))


@pytest.mark.parametrize(
    ("value", "slot_type", "ok"),
    [
        ("parser_1", SlotType.IDENTIFIER, True),
        ("1parser", SlotType.IDENTIFIER, False),
        (3, SlotType.IDENTIFIER, False),
        ("src/a.py", SlotType.PATH, True),
        ("/etc/passwd", SlotType.PATH, False),
        ("a/../b", SlotType.PATH, False),
        ("a//b", SlotType.PATH, False),
        ("a b", SlotType.PATH, False),
        ("any text", SlotType.STRING, True),
        ("line\nbreak", SlotType.STRING, False),
        (5, SlotType.INT, True),
        (True, SlotType.INT, False),
        ("5", SlotType.INT, False),
    ],
)
def test_check_type(value: Any, slot_type: SlotType, ok: bool) -> None:
    assert check_type(value, slot_type) is ok


def test_templates() -> None:
    assert to_snake("ParserModule") == "parser_module"
    assert to_snake("HTTPServer") == "http_server"
    assert to_snake("already-snake") == "already_snake"
    assert parse_template("a{{b}}c") == ["a{b}c"]
    assert escape_literal("{x}") == "{{x}}"
    assert render_template(escape_literal("{x}"), {}) == "{x}"
    with pytest.raises(SkillError, match="unresolved"):
        render_template("{x}", {})


def test_skill_modules_never_eval_exec_or_import_dynamically() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "crystallizer" / "skills"
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, f"{path.name} calls {node.func.id}"
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [alias.name for alias in node.names]
                module = getattr(node, "module", None) or ""
                assert "importlib" not in [module, *names], f"{path.name} imports importlib"


_identifiers = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,20}", fullmatch=True)


@given(name=_identifiers)
def test_executor_output_type_checks(name: str) -> None:
    skill = parse_skill(skill_data())
    actions = instantiate(skill, situation(name=name))
    for action in actions:
        assert Action.model_validate(action.model_dump()) == action
        for value in action.args.values():
            assert isinstance(value, str) or (
                isinstance(value, list) and all(isinstance(v, str) for v in value)
            )
    assert actions[0].args["path"] == f"src/{name}.py"
