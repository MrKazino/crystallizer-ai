"""Ladder integration: skills crystallize across runs, demotion, plugin tiers, report, preview."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crystallizer.approval import ScriptedApprover
from crystallizer.errors import ConfigError
from crystallizer.events import EventKind
from crystallizer.extensions import PluginContext
from crystallizer.models import MockBehavior, MockScript, MockStep
from crystallizer.schemas import (
    Action,
    Event,
    Proposal,
    SkillStatus,
    StepRequest,
    TierResult,
    Usage,
)
from crystallizer.tiers import TierContext
from tests.helpers import act, fast_workspace, greeting_steps, open_harness, task

NAMES = [
    ["alpha", "bravo", "charlie", "delta"],
    ["echo", "foxtrot", "golf", "hotel"],
    ["india", "juliet", "kilo", "lima"],
    ["mike", "november", "oscar", "papa"],
]


def multi_run_script() -> MockScript:
    plans: dict[str, str] = {}
    actions: dict[str, list[MockStep]] = {}
    for run, names in enumerate(NAMES):
        tasks = [task(f"r{run}-{name}", name) for name in names]
        plans[f"goal {run}"] = json.dumps({"tasks": tasks})
        for name in names:
            actions[f"r{run}-{name}"] = greeting_steps(name)
    return MockScript(plans=plans, actions=actions)


def run_once(workspace: Path, script: MockScript, run: int, **kwargs: object) -> object:
    with open_harness(workspace, script, profile="e2e", **kwargs) as harness:  # type: ignore[arg-type]
        harness.plan(f"goal {run}")
        return harness.run()


def test_skills_crystallize_and_take_over(workspace: Path) -> None:
    fast_workspace(workspace)
    script = multi_run_script()
    events: list[Event] = []
    summaries = []
    for run in range(4):
        with open_harness(workspace, script, profile="e2e") as harness:
            harness.bus.subscribe(events.append)
            harness.plan(f"goal {run}")
            summaries.append(harness.run())
            if run == 0:
                assert harness.status().skills == {"candidate": 1}
            if run == 2:
                assert harness.status().skills == {"active": 1}
                preview_skill = harness.skills_list()[0].key
    assert [s.route_mix for s in summaries[:3]] == [{"small": 8}] * 3
    assert summaries[3].route_mix == {"skill": 8}
    assert summaries[3].usage.cost == 0.0
    assert summaries[0].usage.cost > 0
    kinds = {event.kind for event in events}
    assert {EventKind.SKILL_MINED, EventKind.SKILL_PROMOTED, EventKind.ESCALATION} <= kinds
    with open_harness(workspace, script, profile="e2e") as harness:
        entry = harness.registry().get(preview_skill)
        report = harness.report()
    assert entry.live_runs == 4
    assert entry.live_passes == 4
    assert report.by_route["skill"].cost == 0.0
    assert report.estimated_saving > 0
    assert report.skills["active"] == 1


def test_preview_shows_skill_routes(workspace: Path) -> None:
    fast_workspace(workspace)
    script = multi_run_script()
    for run in range(3):
        run_once(workspace, script, run)
    with open_harness(workspace, script, profile="e2e") as harness:
        harness.plan("goal 3")
    with open_harness(workspace, script, profile="e2e", dry_run=True) as harness:
        lines = harness.preview()
    assert {line.route for line in lines} == {"skill"}
    assert lines[0].actions == [act("file_write", path="out/mike.txt", content="mike")]


def import_and_force(workspace: Path, script: MockScript, skill: dict[str, object]) -> str:
    with open_harness(workspace, script, profile="e2e") as harness:
        entry = harness.registry().import_skill(json.dumps(skill))
        assert entry is not None
        harness.skill_promote(entry.key, force=True, reason="test fixture")
        return entry.key


def bad_skill(first_action: dict[str, object]) -> dict[str, object]:
    return {
        "id": "skill-badbadbadbad",
        "version": 1,
        "task_kind": "greet",
        "slots": {"name": {"type": "identifier", "source": "params.name"}},
        "guard": {"all": [{"field": "task_kind", "op": "eq", "value": "greet"}]},
        "steps": [
            {**first_action, "step_kind": "write"},
            {"tool": "file_read", "step_kind": "read", "args": {"path": "out/{name}.txt"}},
        ],
    }


def test_bad_skill_is_demoted_and_task_recovers(workspace: Path) -> None:
    fast_workspace(workspace)
    script = multi_run_script()
    key = import_and_force(
        workspace,
        script,
        bad_skill({"tool": "file_write", "args": {"path": "out/{name}.txt", "content": "WRONG"}}),
    )
    summary = run_once(workspace, script, 0)
    assert summary.exit_code == 0  # type: ignore[attr-defined]
    with open_harness(workspace, script, profile="e2e") as harness:
        entry = harness.registry().get(key)
    assert entry.status is SkillStatus.DEMOTED
    assert entry.live_passes == 0
    assert entry.history[-1].evidence["window_pass_rate"] == 0.0


def test_unsafe_skill_is_demoted_immediately(workspace: Path) -> None:
    fast_workspace(workspace)
    script = multi_run_script()
    key = import_and_force(
        workspace, script, bad_skill({"tool": "file_delete", "args": {"path": "out/{name}.txt"}})
    )
    summary = run_once(workspace, script, 0)
    assert summary.exit_code == 0  # type: ignore[attr-defined]
    assert summary.route_mix == {"small": 8}  # type: ignore[attr-defined]
    with open_harness(workspace, script, profile="e2e") as harness:
        entry = harness.registry().get(key)
    assert entry.status is SkillStatus.DEMOTED
    assert entry.history[-1].evidence == {"unsafe": True}


class OracleTier:
    """Stands in for an external agent system plugged into the ladder."""

    def __init__(self, context: TierContext) -> None:
        self.context = context

    @property
    def name(self) -> str:
        return "oracle"

    def propose(self, request: StepRequest) -> TierResult:
        name = str(request.situation.params["name"])
        if request.situation.step_kind == "write":
            action = Action(tool="file_write", args={"path": f"out/{name}.txt", "content": name})
        else:
            action = Action(tool="file_read", args={"path": f"out/{name}.txt"})
        return TierResult(
            tier=self.name,
            proposal=Proposal(tier=self.name, actions=[action], confidence=0.9),
            usage=Usage(tokens_in=10, tokens_out=5, cost=0.0001, model_calls=1),
        )


class OraclePlugin:
    name = "oracle_plugin"

    def __init__(self, tier_name: str = "oracle") -> None:
        self.tier_name = tier_name

    def register(self, context: PluginContext) -> None:
        context.add_tier(self.tier_name, OracleTier)


def test_plugin_tier_answers_between_models(workspace: Path) -> None:
    fast_workspace(
        workspace, '\n[router]\nladder = ["skill", "small", "oracle", "large", "human"]\n'
    )
    garbage = MockScript(
        plans={"goal": json.dumps({"tasks": [task("t1", "alpha")]})},
        actions={"t1": greeting_steps("alpha", small=MockBehavior.GARBAGE)},
    )
    with open_harness(workspace, garbage, plugins=[OraclePlugin()]) as harness:
        harness.plan("goal")
        summary = harness.run()
    assert summary.exit_code == 0
    assert summary.route_mix == {"oracle": 1, "small": 1}


def test_plugin_tier_misconfiguration(workspace: Path) -> None:
    fast_workspace(workspace, '\n[router]\nladder = ["skill", "oracle2", "human"]\n')
    script = multi_run_script()
    with open_harness(workspace, script, plugins=[OraclePlugin("oracle2")]) as harness:
        harness.plan("goal 0")
        with pytest.raises(ConfigError, match="reports name"):
            harness.run()
    with open_harness(workspace, script) as harness:
        with pytest.raises(ConfigError, match="unknown tier"):
            harness.run()


def test_human_tier_supplies_actions_when_models_fail(workspace: Path) -> None:
    fast_workspace(workspace)
    garbage = MockScript(
        plans={"goal": json.dumps({"tasks": [task("t1", "alpha")]})},
        actions={
            "t1": greeting_steps("alpha", small=MockBehavior.GARBAGE, large=MockBehavior.GARBAGE)
        },
    )
    approver = ScriptedApprover(
        actions=[
            Action(tool="file_write", args={"path": "out/alpha.txt", "content": "alpha"}),
        ]
    )
    with open_harness(workspace, garbage, approver=approver) as harness:
        harness.plan("goal")
        summary = harness.run()
    assert summary.exit_code == 0
    assert summary.route_mix == {"human": 1, "small": 1}


def test_small_disagreement_escalates_to_large(workspace: Path) -> None:
    fast_workspace(workspace)
    script = MockScript(
        plans={"goal": json.dumps({"tasks": [task("t1", "alpha")]})},
        actions={"t1": greeting_steps("alpha", small=MockBehavior.DISAGREE)},
    )
    with open_harness(workspace, script) as harness:
        harness.plan("goal")
        summary = harness.run()
        steps = harness.traces.load_run(summary.run_id or "")
    assert summary.route_mix == {"large": 1, "small": 1}
    escalated = next(step for step in steps if step.route == "large")
    assert escalated.escalation_path == ["skill:no_skill", "small:disagreement"]
    assert escalated.escalation_reason == "disagreement"
    assert escalated.model_calls == 4


def test_cli_report(workspace: Path) -> None:
    import io

    from crystallizer.cli import main

    fast_workspace(workspace)
    script = multi_run_script()
    run_once(workspace, script, 0)
    out = io.StringIO()
    assert main(["--workspace", str(workspace), "report"], out=out) == 0
    assert "all-large baseline" in out.getvalue()
    out = io.StringIO()
    main(["--workspace", str(workspace), "--json", "report"], out=out)
    data = json.loads(out.getvalue())
    assert data["by_route"]["small"]["steps"] == 8
    assert data["saving_pct"] > 0
