"""Miner: templating, mechanical detection, pattern extraction."""

from __future__ import annotations

from crystallizer.schemas import Scalar, Situation
from crystallizer.skills.miner import attempt_groups, mine, templatize
from tests.skills.common import NAMES, act, make_step, module_task, run_steps


def situation(**params: Scalar) -> Situation:
    return Situation(task_kind="k", step_kind="s", params=params)


def test_templatize_basic_and_filters() -> None:
    assert templatize("src/parser.py", situation(name="parser")) == "src/{name}.py"
    assert templatize("tests/test_parser.py", situation(name="parser")) == "tests/test_{name}.py"
    assert templatize("PARSER_ID", situation(name="parser")) == "{name|upper}_ID"
    assert templatize("mod parser_module", situation(name="ParserModule")) == "mod {name|snake}"
    assert templatize("add parsers", situation(name="parser")) == "add parsers"
    assert templatize("ratio", situation(name="rat")) == "ratio"
    assert templatize("a {x} b", situation(name="xyz")) == "a {{x}} b"
    assert templatize("id-1234", situation(n=1234)) == "id-{n}"
    assert templatize("io", situation(name="io")) == "io"
    assert templatize("new_total total", situation(old="total", new="new_total")) == "{new} {old}"


def test_templatize_falls_back_when_round_trip_fails() -> None:
    # "Ab" -> lower form "ab" also appears; the template must still round-trip exactly.
    value = "AbC abc"
    assert templatize(value, situation(name="AbC")) in {"{name} {name|lower}", value}


def test_mine_finds_mechanical_runs() -> None:
    steps = run_steps("run-1", NAMES[:4])
    patterns = mine(steps, min_repeats=2)
    kinds = sorted([step.step_kind for step in p.steps] for p in patterns)
    assert kinds == [["scaffold"], ["stage", "commit"]]
    for pattern in patterns:
        assert pattern.support == 4
        assert pattern.slot_fields == ["name"]
        assert pattern.runs == ["run-1"]
        assert len(pattern.provenance) == 4 * len(pattern.steps)
    commit = next(p for p in patterns if p.steps[0].step_kind == "stage")
    assert commit.steps[1].args["argv"] == ["git", "commit", "-m", "add {name}"]
    assert commit.occurrences[0].start == 2


def test_mine_respects_min_repeats_and_eligibility() -> None:
    steps = run_steps("run-1", NAMES[:2])
    assert mine(steps, min_repeats=3) == []
    unverified = [s.model_copy(update={"verified": False}) for s in run_steps("run-1", NAMES[:4])]
    assert mine(unverified, min_repeats=2) == []
    skill_routed = [s.model_copy(update={"route": "skill"}) for s in run_steps("run-1", NAMES[:4])]
    assert mine(skill_routed, min_repeats=2) == []
    redacted = [s.model_copy(update={"redacted": True}) for s in run_steps("run-1", NAMES[:4])]
    assert mine(redacted, min_repeats=2) == []


def test_varying_non_derivable_arguments_are_not_mechanical() -> None:
    steps = [
        step
        for index, name in enumerate(NAMES[:4])
        for step in module_task("run-1", name, "body", commit_message=f"commit #{index}")
    ]
    patterns = mine(steps, min_repeats=2)
    kinds = sorted([step.step_kind for step in p.steps] for p in patterns)
    # The constant body is mechanical here; the commit message is not derivable, so the run
    # scaffold→implement→stage stops before commit.
    assert kinds == [["scaffold", "implement", "stage"]]


def test_replayed_duplicates_and_gaps() -> None:
    steps = run_steps("run-1", NAMES[:3])
    replayed = [s.model_copy(update={"replayed": True}) for s in steps]
    assert len(attempt_groups(steps + replayed)[("run-1", "add-parser", 1)]) == 4
    gap = [s for s in steps if s.index != 2]  # stage missing breaks the stage→commit run
    patterns = mine(gap, min_repeats=2)
    kinds = sorted([step.step_kind for step in p.steps] for p in patterns)
    assert kinds == [["commit"], ["scaffold"]]


def test_multiple_runs_add_support() -> None:
    steps = run_steps("run-1", NAMES[:2]) + run_steps("run-2", NAMES[2:4], seed=1)
    patterns = mine(steps, min_repeats=3)
    assert {p.support for p in patterns} == {4}
    assert patterns[0].runs == ["run-1", "run-2"]


def test_failed_steps_are_not_evidence() -> None:
    step = make_step("r", "t", 0, "k", act("file_read", path="a"), {"name": "abc"}, ok=False)
    assert attempt_groups([step]) == {}
