"""Reproducible benchmark: run a scenario N times with MockModel, keeping skills between runs.

Each run gets a fresh workspace (scenario fixtures committed to a new git repository with a local
identity and signing disabled), while the state directory (skills, traces, memory, journal)
persists across runs. The plan comes from the scenario and is validated as a DAG, so planning cost
does not blur the per-run comparison. Output is deterministic for a given scenario seed.

What this proves: the routing, mining, shadow-testing and promotion mechanics work end to end.
What it does not prove: cost savings with a real model (MockModel token counts are scripted).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from crystallizer.api import Harness
from crystallizer.approval import DenyApprover
from crystallizer.clock import FixedClock, make_rng
from crystallizer.errors import CrystallizerError
from crystallizer.models import MockModel
from crystallizer.scenario import Scenario, instantiate
from crystallizer.schemas import BenchReport, BenchRun, BenchSummary, TaskState

GIT_IDENTITY = (
    ("user.name", "crystallizer-bench"),
    ("user.email", "bench@crystallizer.invalid"),
    ("commit.gpgsign", "false"),
    ("init.defaultBranch", "main"),
)


def _git(workspace: Path, env: Mapping[str, str], *args: str) -> None:
    program = shutil.which("git", path=env.get("PATH"))
    if program is None:
        raise CrystallizerError("git is required to prepare benchmark workspaces")
    subprocess.run(
        [program, *args],
        cwd=workspace,
        env=dict(env),
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )


def prepare_workspace(workspace: Path, fixtures: dict[str, str], env: Mapping[str, str]) -> None:
    """Write fixtures and commit them to a fresh repository (internal setup, not agent tools)."""
    workspace.mkdir(parents=True, exist_ok=True)
    for relative, content in sorted(fixtures.items()):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(workspace, env, "init", "-q")
    for key, value in GIT_IDENTITY:
        _git(workspace, env, "config", key, value)
    _git(workspace, env, "add", "-A")
    _git(workspace, env, "commit", "-q", "--allow-empty", "-m", "fixtures")


def run_bench(
    scenario: Scenario, runs: int, root: Path, *, profile: str | None = None
) -> BenchReport:
    """Run ``scenario`` ``runs`` times under ``root`` and return the report."""
    if runs < 1:
        raise CrystallizerError("--runs must be at least 1")
    state_dir = (root / "state").resolve()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root.resolve()),
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    rng = make_rng(scenario.seed)
    clock = FixedClock()
    used: set[int] = set()
    results: list[BenchRun] = []
    for number in range(1, runs + 1):
        spec = instantiate(scenario, rng, used)
        workspace = root / f"run-{number}"
        prepare_workspace(workspace, spec.fixtures, env)
        (workspace / "crystallizer.toml").write_text(
            f'[workspace]\nstate_dir = "{state_dir.as_posix()}"\n', encoding="utf-8"
        )
        model = MockModel(spec.script, seed=scenario.seed + number)
        with Harness.open(
            workspace,
            profile=profile,
            clock=clock,
            seed=scenario.seed + number,
            model=model,
            approver=DenyApprover(),
            environ=env,
        ) as harness:
            harness.adopt_plan(spec.plan)
            summary = harness.run()
            skills = harness.registry().counts()
        done = sum(1 for task in summary.tasks if task.state is TaskState.DONE)
        results.append(
            BenchRun(
                run=number,
                run_id=summary.run_id or "",
                cost=round(summary.usage.cost, 9),
                tokens_in=summary.usage.tokens_in,
                tokens_out=summary.usage.tokens_out,
                model_calls=summary.usage.model_calls,
                route_mix=summary.route_mix,
                skills=skills,
                acceptance_pass_rate=round(done / len(summary.tasks), 6) if summary.tasks else 0.0,
                tasks=len(summary.tasks),
                steps=summary.steps,
                mechanical_steps=spec.mechanical_steps,
                skill_steps=summary.route_mix.get("skill", 0),
                exit_code=summary.exit_code,
            )
        )
    first, last = results[0].cost, results[-1].cost
    active_runs = [r.run for r in results if r.skills.get("active", 0) > 0 and r.skill_steps > 0]
    return BenchReport(
        scenario=scenario.name,
        description=scenario.description,
        seed=scenario.seed,
        profile=profile,
        runs=results,
        summary=BenchSummary(
            cost_first=first,
            cost_last=last,
            cost_ratio=round(last / first, 6) if first else None,
            first_run_with_active_skill=active_runs[0] if active_runs else None,
            acceptance_pass_rate=round(
                sum(r.acceptance_pass_rate for r in results) / len(results), 6
            ),
            total_cost=round(sum(r.cost for r in results), 9),
        ),
    )
