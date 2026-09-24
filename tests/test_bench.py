"""Scenario loading and instantiation, the bench harness, and the bench CLI."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

from crystallizer.bench import run_bench
from crystallizer.cli import main, scenario_dirs
from crystallizer.clock import make_rng
from crystallizer.errors import ConfigError, CrystallizerError
from crystallizer.scenario import find_scenario, instantiate, load_scenario

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.mark.parametrize("name", ["add-modules", "fix-tests", "rename-refactor"])
def test_scenarios_are_valid_and_deterministic(name: str) -> None:
    scenario = load_scenario(SCENARIOS / f"{name}.toml")
    assert scenario.tasks_per_run == 4
    first = instantiate(scenario, make_rng(scenario.seed), set())
    second = instantiate(scenario, make_rng(scenario.seed), set())
    assert first == second
    assert first.total_steps == 16 or name == "add-modules"
    assert first.mechanical_steps == (16 if name == "add-modules" else 12)
    disagreements = sum(
        1
        for steps in first.script.actions.values()
        for step in steps
        if step.small.value == "disagree"
    )
    assert disagreements >= 2
    used: set[int] = set()
    rng = make_rng(1)
    runs = [instantiate(scenario, rng, used) for _ in range(3)]
    assert len(used) == 12  # unique model-authored values across runs
    names = [tuple(t.params.values()) for run in runs for t in run.plan.tasks]
    assert len(set(names)) > 4  # parameters genuinely vary between runs


def test_find_and_invalid_scenarios(tmp_path: Path) -> None:
    assert find_scenario("add-modules", [tmp_path, SCENARIOS]) == SCENARIOS / "add-modules.toml"
    assert find_scenario(str(SCENARIOS / "fix-tests.toml"), []) == SCENARIOS / "fix-tests.toml"
    with pytest.raises(ConfigError, match="not found"):
        find_scenario("nope", [tmp_path])
    bad = tmp_path / "bad.toml"
    bad.write_text("name = [", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid scenario"):
        load_scenario(bad)
    text = (SCENARIOS / "add-modules.toml").read_text(encoding="utf-8")
    small_pool = tmp_path / "small.toml"
    small_pool.write_text(text.replace("tasks_per_run = 4", "tasks_per_run = 40"), encoding="utf-8")
    with pytest.raises(ConfigError, match="distinct values"):
        load_scenario(small_pool)


def test_rename_refactor_single_run(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIOS / "rename-refactor.toml")
    report = run_bench(scenario, 1, tmp_path)
    run = report.runs[0]
    assert run.acceptance_pass_rate == 1.0
    assert run.route_mix == {"large": 4, "small": 12}
    assert report.summary.cost_ratio == 1.0
    assert report.summary.first_run_with_active_skill is None
    notes = sorted(p.name for p in (tmp_path / "run-1" / "notes").iterdir())
    assert len(notes) == 4


def test_bench_requires_runs_and_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = load_scenario(SCENARIOS / "fix-tests.toml")
    with pytest.raises(CrystallizerError, match="at least 1"):
        run_bench(scenario, 0, tmp_path)
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
    with pytest.raises(CrystallizerError, match="git is required"):
        run_bench(scenario, 1, tmp_path)


def cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(list(argv), out=out), out.getvalue()


def test_bench_cli(tmp_path: Path) -> None:
    code, out = cli("bench", "--scenario", "fix-tests", "--runs", "1", "--dry-run")
    assert code == 0
    assert "would run fix-tests 1 time(s)" in out
    keep = tmp_path / "keep"
    code, out = cli("bench", "--scenario", "fix-tests", "--runs", "1", "--keep", str(keep))
    assert code == 0
    report = json.loads(out)
    assert report["scenario"] == "fix-tests"
    assert (keep / "state" / "traces").is_dir()
    code, out = cli("--profile", "e2e", "bench", "--scenario", "fix-tests", "--runs", "1")
    assert json.loads(out)["profile"] == "e2e"
    assert cli("bench", "--scenario", "missing-scenario")[0] == 2


def test_scenario_dirs_are_unique(tmp_path: Path) -> None:
    dirs = scenario_dirs(tmp_path)
    assert len({d.resolve() for d in dirs}) == len(dirs)
    assert SCENARIOS.resolve() in {d.resolve() for d in dirs}


def test_global_flags_after_subcommand(tmp_path: Path) -> None:
    code, out = cli("status", "--workspace", str(tmp_path), "--json")
    assert code == 0
    assert json.loads(out)["has_plan"] is False
