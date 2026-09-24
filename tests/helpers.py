"""Builders for scripted MockModel projects used across tests."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crystallizer.api import Harness
from crystallizer.approval import Approver, DenyApprover
from crystallizer.clock import FixedClock
from crystallizer.extensions import Plugin
from crystallizer.faults import NO_FAULTS, FaultInjector
from crystallizer.models import MockBehavior, MockModel, MockScript, MockStep
from crystallizer.schemas import Action, ArgValue

GOAL = "write greeting files"
FAST_CONFIG = """\
[policy]
allow_irreversible = ["shell:python"]
"""


def act(tool: str, **args: ArgValue) -> Action:
    """Shorthand for an action."""
    return Action(tool=tool, args=args)


def check_file(path: str, content: str) -> list[str]:
    """A fast acceptance command asserting a file's content (allow-listed python -c)."""
    code = (
        f"import pathlib,sys; "
        f"sys.exit(0 if pathlib.Path({path!r}).read_text() == {content!r} else 1)"
    )
    return ["python", "-c", code]


def task(
    task_id: str,
    name: str,
    *,
    depends_on: Sequence[str] = (),
    steps: Sequence[str] = ("write", "read"),
    kind: str = "greet",
) -> dict[str, Any]:
    """A plan task that writes ``out/<name>.txt``."""
    return {
        "id": task_id,
        "title": f"Write greeting {name}",
        "kind": kind,
        "params": {"name": name},
        "depends_on": list(depends_on),
        "steps": [{"kind": step, "description": f"{step} {name}"} for step in steps],
        "acceptance_commands": [check_file(f"out/{name}.txt", name)],
    }


def greeting_steps(
    name: str,
    small: MockBehavior = MockBehavior.CORRECT,
    large: MockBehavior = MockBehavior.CORRECT,
) -> list[MockStep]:
    """Scripted steps for :func:`task`: write the file, then read it back."""
    return [
        MockStep(
            action=act("file_write", path=f"out/{name}.txt", content=name), small=small, large=large
        ),
        MockStep(action=act("file_read", path=f"out/{name}.txt")),
    ]


def project_script(extra: dict[str, list[MockStep]] | None = None) -> MockScript:
    """Two tasks: ``t1`` writes alpha, ``t2`` (after t1) writes beta."""
    plan = {"tasks": [task("t1", "alpha"), task("t2", "beta", depends_on=["t1"])]}
    actions = {"t1": greeting_steps("alpha"), "t2": greeting_steps("beta")}
    actions.update(extra or {})
    return MockScript(plans={GOAL: "Here is the plan:\n" + json.dumps(plan)}, actions=actions)


def open_harness(
    workspace: Path,
    script: MockScript,
    *,
    approver: Approver | None = None,
    faults: FaultInjector = NO_FAULTS,
    profile: str | None = None,
    plugins: Sequence[Plugin] = (),
    seed: int = 0,
    dry_run: bool = False,
    model: MockModel | None = None,
) -> Harness:
    """Open a harness on ``workspace`` with a deterministic clock and a scripted model."""
    return Harness.open(
        workspace,
        profile=profile,
        clock=FixedClock(),
        seed=seed,
        model=model or MockModel(script, seed),
        approver=approver or DenyApprover(),
        faults=faults,
        plugins=plugins,
        environ={"PATH": os.environ.get("PATH", ""), "HOME": str(workspace)},
        dry_run=dry_run,
    )


def fast_workspace(workspace: Path, extra_config: str = "") -> Path:
    """Write a config that allow-lists ``python -c`` acceptance checks."""
    (workspace / "crystallizer.toml").write_text(FAST_CONFIG + extra_config, encoding="utf-8")
    return workspace
