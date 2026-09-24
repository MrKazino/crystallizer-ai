"""Compiler, shadow testing and the registry lifecycle on recorded traces."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crystallizer.cli import main
from crystallizer.clock import FixedClock
from crystallizer.config import PolicyConfig, SkillsConfig
from crystallizer.db import Database
from crystallizer.errors import SkillError, UsageError
from crystallizer.events import EventBus
from crystallizer.policy import Policy
from crystallizer.redaction import Redactor
from crystallizer.schemas import GuardOp, Scalar, Situation, SkillStatus, SlotType
from crystallizer.skills.compiler import (
    build_guard,
    compile_pattern,
    content_hash,
    guard_summary,
    infer_slot_type,
)
from crystallizer.skills.executor import evaluate_guard, instantiate
from crystallizer.skills.miner import Occurrence, Pattern, mine
from crystallizer.skills.registry import SkillRegistry, load_manifest, wilson_lower
from crystallizer.skills.shadow import shadow_evaluate
from crystallizer.traces import TraceStore
from tests.skills.common import NAMES, act, record, run_steps

E2E = SkillsConfig(min_repeats=2, min_shadow_runs=5)


def registry(state_dir: Path, db: Database, config: SkillsConfig = E2E) -> SkillRegistry:
    return SkillRegistry(
        state_dir,
        db,
        config,
        FixedClock(),
        Policy(PolicyConfig()),
        EventBus(FixedClock(), Redactor()),
    )


def patterns_for(names: list[str]) -> list[Pattern]:
    return mine(run_steps("run-1", names), min_repeats=2)


# --------------------------------------------------------------------------- compiler


def test_compile_pattern_guard_and_identity() -> None:
    pattern = next(p for p in patterns_for(NAMES[:4]) if p.steps[0].step_kind == "stage")
    skill = compile_pattern(pattern)
    assert skill is not None
    assert skill.id.startswith("skill-")
    assert skill.slots["name"].type is SlotType.IDENTIFIER
    assert skill.slots["name"].source == "params.name"
    ops = {(c.field, c.op) for c in skill.guard.all}
    assert ("task_kind", GuardOp.EQ) in ops
    assert ("step_kind", GuardOp.EQ) in ops
    assert ("step_index", GuardOp.RANGE) in ops
    assert ("params.name", GuardOp.HAS_TYPE) in ops
    again = compile_pattern(
        next(p for p in patterns_for(NAMES[4:8]) if p.steps[0].step_kind == "stage")
    )
    assert again is not None
    assert again.id == skill.id
    assert content_hash(again) == content_hash(skill)
    actions = instantiate(
        skill,
        Situation(task_kind="add_module", step_kind="stage", step_index=2, params={"name": "zeta"}),
    )
    assert actions[1].args["argv"] == ["git", "commit", "-m", "add zeta"]
    assert "params.name is identifier" in guard_summary(skill.guard)


def occurrence(params: dict[str, Scalar], index: int = 0) -> Occurrence:
    return Occurrence(
        run_id="r",
        task_id="t",
        attempt=1,
        start=index,
        step_ids=[],
        situation=Situation(task_kind="k", step_kind="s", step_index=index, params=params),
    )


def test_build_guard_covers_all_param_kinds() -> None:
    base = patterns_for(NAMES[:2])[0]
    pattern = base.model_copy(
        update={
            "task_kind": "k",
            "slot_fields": [],
            "occurrences": [
                occurrence({"lang": "py", "size": 1, "flag": True, "mix": 1, "opt": "a"}, 0),
                occurrence({"lang": "go", "size": 5, "flag": True, "mix": "x"}, 2),
            ],
        }
    )
    guard = build_guard(pattern, {})
    by_field = {c.field: c for c in guard.all}
    assert by_field["params.lang"].op is GuardOp.IN
    assert by_field["params.lang"].value == ["go", "py"]
    assert by_field["params.size"].op is GuardOp.RANGE
    assert by_field["params.flag"].op is GuardOp.EQ
    assert by_field["params.mix"].op is GuardOp.EXISTS
    assert "params.opt" not in by_field
    assert by_field["step_index"].op is GuardOp.RANGE
    summary = guard_summary(guard)
    assert "params.lang in" in summary
    assert "params.mix exists" in summary
    assert "<=" in summary


def test_infer_slot_type_and_uncompilable_patterns() -> None:
    assert infer_slot_type([1, 2]) is SlotType.INT
    assert infer_slot_type(["a", "b_1"]) is SlotType.IDENTIFIER
    assert infer_slot_type(["src/a.py", "b"]) is SlotType.PATH
    assert infer_slot_type(["a b", "c"]) is SlotType.STRING
    assert infer_slot_type(["a", 1]) is None
    assert infer_slot_type([]) is None
    base = patterns_for(NAMES[:2])[0]
    mixed = base.model_copy(
        update={"occurrences": [occurrence({"name": "a"}), occurrence({"name": 3})]}
    )
    assert compile_pattern(mixed) is None
    missing = base.model_copy(update={"occurrences": [occurrence({"name": "a"}), occurrence({})]})
    assert compile_pattern(missing) is None


_value = st.one_of(st.sampled_from(["py", "go", "rs"]), st.integers(0, 10))


@given(
    names=st.lists(st.sampled_from(NAMES), min_size=2, max_size=4, unique=True),
    lang=st.sampled_from(["py", "go"]),
    size=st.integers(0, 10),
    outside=st.sampled_from(["task_kind", "step_kind", "step_index", "lang", "size", "name"]),
)
def test_guard_never_matches_outside_mined_domain(
    names: list[str], lang: str, size: int, outside: str
) -> None:
    pattern = patterns_for(NAMES[:4])[0]
    occurrences = [
        occurrence({"name": name, "lang": lang, "size": size + i}, pattern.occurrences[0].start)
        for i, name in enumerate(names)
    ]
    pattern = pattern.model_copy(update={"occurrences": occurrences, "task_kind": "k"})
    skill = compile_pattern(pattern)
    assert skill is not None
    for occ in occurrences:
        assert evaluate_guard(
            skill.guard, occ.situation.model_copy(update={"step_kind": pattern.steps[0].step_kind})
        )
    params: dict[str, Scalar] = {"name": "fresh_name", "lang": lang, "size": size}
    fields: dict[str, Any] = {
        "task_kind": "k",
        "step_kind": pattern.steps[0].step_kind,
        "step_index": pattern.occurrences[0].start,
        "params": params,
    }
    if outside == "task_kind":
        fields["task_kind"] = "other"
    elif outside == "step_kind":
        fields["step_kind"] = "other"
    elif outside == "step_index":
        fields["step_index"] = pattern.occurrences[0].start + 1
    elif outside == "lang":
        params["lang"] = "rs"
    elif outside == "size":
        params["size"] = size + len(names) + 5
    else:
        params["name"] = "not an identifier!"
    assert not evaluate_guard(skill.guard, Situation.model_validate(fields))


# --------------------------------------------------------------------------- shadow


def compiled_commit_skill() -> tuple[Any, Pattern]:
    pattern = next(p for p in patterns_for(NAMES[:4]) if p.steps[0].step_kind == "stage")
    skill = compile_pattern(pattern)
    assert skill is not None
    return skill, pattern


def test_shadow_pass_fail_and_exclusions() -> None:
    skill, pattern = compiled_commit_skill()
    policy = Policy(PolicyConfig())
    later = run_steps("run-2", NAMES[4:6], seed=2)
    task = [s for s in later if s.task_id == "add-loader"]
    obs = shadow_evaluate(
        skill, task, provenance=set(pattern.provenance), derived_runs={"run-1"}, policy=policy
    )
    assert [(o.passed, o.unsafe) for o in obs] == [(True, False)]
    assert obs[0].occurrence == "run-2:add-loader:1:2"
    assert (
        shadow_evaluate(skill, task, provenance=set(), derived_runs={"run-2"}, policy=policy) == []
    )
    assert (
        shadow_evaluate(skill, task, provenance={task[2].id}, derived_runs=set(), policy=policy)
        == []
    )
    redacted = [s.model_copy(update={"redacted": True}) for s in task]
    assert (
        shadow_evaluate(skill, redacted, provenance=set(), derived_runs=set(), policy=policy) == []
    )
    other = [
        s.model_copy(update={"action": act("shell", argv=["git", "commit", "-m", "other"])})
        if s.index == 3
        else s
        for s in task
    ]
    assert [
        o.passed
        for o in shadow_evaluate(skill, other, provenance=set(), derived_runs=set(), policy=policy)
    ] == [False]


def test_shadow_unsafe_diff() -> None:
    skill, _ = compiled_commit_skill()
    unsafe = skill.model_copy(
        update={
            "steps": [
                skill.steps[0],
                skill.steps[1].model_copy(
                    update={"tool": "file_delete", "args": {"path": "src/{name}.py"}}
                ),
            ]
        }
    )
    task = [s for s in run_steps("run-2", ["loader"]) if s.task_id == "add-loader"]
    obs = shadow_evaluate(
        unsafe, task, provenance=set(), derived_runs=set(), policy=Policy(PolicyConfig())
    )
    assert [(o.passed, o.unsafe) for o in obs] == [(False, True)]


# --------------------------------------------------------------------------- registry


def test_registry_lifecycle_on_recorded_traces(state_dir: Path, db: Database) -> None:
    traces = state_dir / "traces"
    record(traces, run_steps("run-1", NAMES[:4]))
    store = TraceStore(traces)
    reg = registry(state_dir, db)
    added = reg.mine(store)
    assert len(added) == 2
    assert reg.counts() == {"candidate": 2, "active": 0, "demoted": 0}
    assert reg.mine(store) == []  # no churn on re-mining
    assert reg.try_promote_all() == []  # no shadow evidence yet
    record(traces, run_steps("run-2", NAMES[4:8], seed=1))
    record(traces, run_steps("run-3", NAMES[:4], seed=2))
    assert reg.evaluate(store) == 16
    assert reg.evaluate(store) == 0  # idempotent
    promoted = reg.try_promote_all()
    assert len(promoted) == 2
    entry = promoted[0]
    assert entry.status is SkillStatus.ACTIVE
    assert entry.shadow_runs == 8
    assert entry.pass_rate == 1.0
    assert entry.history[-1].event == "promoted"
    assert entry.history[-1].evidence["wilson_lower_95"] == wilson_lower(8, 8)
    assert (state_dir / "skills" / entry.source_file).is_file()
    assert not (state_dir / "skills" / "candidates" / Path(entry.source_file).name).exists()
    pairs = reg.active_skills()
    assert len(pairs) == 2
    assert len(pairs[0][0].steps) >= len(pairs[1][0].steps)

    key = entry.key
    assert reg.record_live(key, "run-4:t:1:0", True) is None
    assert reg.record_live(key, "run-4:t:1:0", False) is None  # duplicate occurrence ignored
    demoted = reg.record_live(key, "run-4:t2:1:0", False)
    assert demoted is not None
    assert demoted.status is SkillStatus.DEMOTED
    assert demoted.history[-1].evidence["window_pass_rate"] == 0.5
    assert reg.mine(store) == []  # same content as a demoted version: never re-added
    manifest = load_manifest(state_dir / "skills")
    assert manifest.registry_version == reg.version


def test_manual_promotion_rules(state_dir: Path, db: Database) -> None:
    traces = state_dir / "traces"
    record(traces, run_steps("run-1", NAMES[:4]))
    reg = registry(state_dir, db)
    first = reg.mine(TraceStore(traces))[0]
    with pytest.raises(SkillError, match="refused"):
        reg.promote(first.id)
    with pytest.raises(UsageError, match="reason"):
        reg.promote(first.id, force=True)
    forced = reg.promote(first.id, force=True, reason="reviewed by hand")
    assert forced.history[-1].forced
    assert forced.history[-1].reason == "forced: reviewed by hand"
    with pytest.raises(SkillError, match="only candidates"):
        reg.promote(first.key)
    demoted = reg.demote(first.key, reason="manual")
    assert demoted.status is SkillStatus.DEMOTED
    with pytest.raises(SkillError, match="only active"):
        reg.demote(first.key, reason="again")
    with pytest.raises(UsageError, match="unknown skill"):
        reg.get("skill-missing")


def test_unsafe_demotion_and_import_export(state_dir: Path, db: Database, tmp_path: Path) -> None:
    traces = state_dir / "traces"
    record(traces, run_steps("run-1", NAMES[:4]))
    reg = registry(state_dir, db)
    first = reg.mine(TraceStore(traces))[0]
    reg.promote(first.id, force=True, reason="test")
    demoted = reg.demote_unsafe(first.id, "emitted an irreversible action")
    assert demoted.history[-1].evidence == {"unsafe": True}
    exported = reg.export_skill(first.id)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    with Database(other_dir / "state.db") as other_db:
        other = registry(other_dir, other_db)
        imported = other.import_skill(json.dumps(exported))
        assert imported is not None
        assert imported.origin == "imported"
        assert imported.status is SkillStatus.CANDIDATE
        assert imported.history[0].event == "imported"
        assert other.import_skill(json.dumps(exported)) is None
        with pytest.raises(SkillError):
            other.import_skill('{"id": "skill-x"}')


def test_changed_pattern_for_demoted_id_becomes_new_version(state_dir: Path, db: Database) -> None:
    reg = registry(state_dir, db)
    skill, pattern = compiled_commit_skill()
    first = reg.add_candidate(skill, provenance=pattern.provenance, derived_runs=["run-1"])
    assert first is not None
    reg.promote(first.key, force=True, reason="t")
    reg.demote(first.key, reason="t")
    widened = skill.model_copy(
        update={"guard": skill.guard.model_copy(update={"all": skill.guard.all[:-1]})}
    )
    second = reg.add_candidate(widened)
    assert second is not None
    assert second.version == 2
    assert reg.get(skill.id).version == 2
    assert reg.get(f"{skill.id}@v1").status is SkillStatus.DEMOTED


def test_wilson_lower() -> None:
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(5, 5) == pytest.approx(0.565509, abs=1e-6)
    assert wilson_lower(20, 20) > wilson_lower(5, 5)


# --------------------------------------------------------------------------- CLI


def cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(list(argv), out=out), out.getvalue()


def test_skills_cli(workspace: Path, tmp_path: Path) -> None:
    (workspace / "crystallizer.toml").write_text("", encoding="utf-8")
    ws = str(workspace)
    record(workspace / ".crystallizer" / "traces", run_steps("run-1", NAMES[:4]))
    assert "no skills yet" in cli("--workspace", ws, "skills", "list")[1]
    assert "dry run: would mine" in cli("--workspace", ws, "--dry-run", "skills", "mine")[1]
    code, out = cli("--workspace", ws, "--profile", "e2e", "skills", "mine")
    assert code == 0
    assert out.count("candidate") == 2
    entries = json.loads(cli("--workspace", ws, "--json", "skills", "list")[1])
    ref = entries[0]["id"]
    shown = json.loads(cli("--workspace", ws, "skills", "show", ref)[1])
    assert shown["manifest"]["id"] == ref
    exported = cli("--workspace", ws, "skills", "export", ref)[1]
    assert json.loads(exported)["id"] == ref
    assert cli("--workspace", ws, "skills", "promote", ref)[0] == 1
    assert cli("--workspace", ws, "skills", "promote", ref, "--force")[0] == 2
    code, out = cli("--workspace", ws, "skills", "promote", ref, "--force", "--reason", "ok")
    assert code == 0
    assert "promoted" in out
    assert "demoted" in cli("--workspace", ws, "skills", "demote", ref, "--reason", "done")[1]
    assert "0 new shadow" in cli("--workspace", ws, "--profile", "e2e", "skills", "evaluate")[1]
    other = tmp_path / "other"
    other.mkdir()
    skill_file = tmp_path / "skill.json"
    skill_file.write_text(exported, encoding="utf-8")
    assert "imported" in cli("--workspace", str(other), "skills", "import", str(skill_file))[1]
    assert "already known" in cli("--workspace", str(other), "skills", "import", str(skill_file))[1]
    status = json.loads(cli("--workspace", ws, "--json", "status")[1])
    assert status["skills"] == {"candidate": 1, "demoted": 1}
