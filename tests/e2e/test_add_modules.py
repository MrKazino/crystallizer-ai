"""End-to-end: the add-modules scenario, five runs under the e2e profile (spec section 11)."""

from __future__ import annotations

from pathlib import Path

from crystallizer.bench import run_bench
from crystallizer.scenario import load_scenario
from crystallizer.traces import TraceStore

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios"


def test_add_modules_gets_cheaper(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIOS / "add-modules.toml")
    report = run_bench(scenario, 5, tmp_path, profile="e2e")
    runs = report.runs
    assert len(runs) == 5
    # At least one skill promoted by run 4.
    assert runs[3].skills["active"] >= 1
    # Mined after run 1, shadow-tested in runs 2 and 3, active by run 4.
    assert runs[0].skills["candidate"] >= 1
    assert runs[1].skills["active"] == 0
    assert report.summary.first_run_with_active_skill == 4
    # The cost of run 5 is at most 50% of run 1.
    assert runs[4].cost <= 0.5 * runs[0].cost
    # Acceptance checks pass in every run.
    assert all(run.acceptance_pass_rate == 1.0 for run in runs)
    assert all(run.exit_code == 0 for run in runs)
    # The trace log of run 5 contains skill-routed steps.
    steps = TraceStore(tmp_path / "state" / "traces").load_run(runs[4].run_id)
    assert any(step.route == "skill" for step in steps)
    # Every step the scenario marks mechanical was crystallized; nothing else was.
    assert runs[4].skill_steps == runs[4].mechanical_steps == 16
    assert runs[4].route_mix == {"skill": 16, "small": 4}
