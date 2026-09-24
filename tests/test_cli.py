"""CLI behavior and exit codes."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crystallizer.cli import main


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def test_schemas_export_and_check(tmp_path: Path) -> None:
    directory = tmp_path / "schemas"
    code, _ = run_cli("schemas", "check", "--dir", str(directory))
    assert code == 1
    code, out = run_cli("--json", "schemas", "export", "--dir", str(directory))
    assert code == 0
    assert "plan.schema.json" in json.loads(out)["written"]
    code, out = run_cli("schemas", "check", "--dir", str(directory))
    assert code == 0
    assert "up to date" in out


def test_usage_error_exit_code_2() -> None:
    with pytest.raises(SystemExit) as info:
        main(["no-such-command"])
    assert info.value.code == 2


def test_module_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "crystallizer", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "crystallizer" in completed.stdout


def cli_workspace(workspace: Path) -> Path:
    from tests.helpers import FAST_CONFIG, project_script

    (workspace / "script.json").write_text(project_script().model_dump_json(), encoding="utf-8")
    (workspace / "crystallizer.toml").write_text(
        FAST_CONFIG + '\n[model]\nmock_script = "script.json"\n', encoding="utf-8"
    )
    return workspace


def test_cli_full_flow(workspace: Path) -> None:
    from tests.helpers import GOAL

    ws = str(cli_workspace(workspace))
    code, out = run_cli("--workspace", ws, "init")
    assert code == 0
    assert "created" in out
    code, out = run_cli("--workspace", ws, "status")
    assert "no plan yet" in out
    code, out = run_cli("--workspace", ws, "--dry-run", "plan", GOAL)
    assert "would call large model" in out
    code, out = run_cli("--workspace", ws, "plan", GOAL)
    assert code == 0
    assert "t1: Write greeting alpha" in out
    code, out = run_cli("--workspace", ws, "--dry-run", "run")
    assert "would call small model" in out
    code, out = run_cli("--workspace", ws, "--json", "run")
    assert code == 0
    assert json.loads(out)["route_mix"] == {"small": 4}
    code, out = run_cli("--workspace", ws, "status")
    assert "done" in out
    code, out = run_cli("--workspace", ws, "resume")
    assert "nothing to resume" in out
    code, out = run_cli("--workspace", ws, "run")
    assert code == 0
    assert "steps=0" in out
    code, out = run_cli("--workspace", ws, "--dry-run", "run")
    assert "nothing to run" in out
    code, out = run_cli("--workspace", ws, "--json", "--dry-run", "init")
    assert json.loads(out)["dry_run"] is True


def test_cli_tools_list(workspace: Path) -> None:
    code, out = run_cli("--workspace", str(workspace), "tools", "list")
    assert code == 0
    assert "file_delete" in out
    assert "irreversible" in out
    code, out = run_cli("--workspace", str(workspace), "--json", "tools", "list")
    assert {spec["name"] for spec in json.loads(out)} >= {"shell", "file_write"}


def test_cli_error_exit_codes(workspace: Path, tmp_path: Path) -> None:
    code, _ = run_cli("--workspace", str(workspace), "run")
    assert code == 2
    code, _ = run_cli("--workspace", str(tmp_path / "missing"), "status")
    assert code == 2
    (workspace / "crystallizer.toml").write_text("[nope]\n", encoding="utf-8")
    code, _ = run_cli("--workspace", str(workspace), "status")
    assert code == 2


def test_cli_task_failure_exit_3(workspace: Path) -> None:
    from crystallizer.models import MockBehavior
    from tests.helpers import GOAL, greeting_steps, project_script

    cli_workspace(workspace)
    wrong = greeting_steps("alpha", small=MockBehavior.WRONG, large=MockBehavior.WRONG)
    script = project_script({"t1": wrong})
    (workspace / "script.json").write_text(script.model_dump_json(), encoding="utf-8")
    ws = str(workspace)
    assert run_cli("--workspace", ws, "plan", GOAL)[0] == 0
    code, out = run_cli("--workspace", ws, "run")
    assert code == 3
    assert "failed" in out


def test_cli_unexpected_error_is_redacted(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    from crystallizer import api

    def boom(self: object) -> None:
        raise RuntimeError("leak token=hunter2")

    monkeypatch.setattr(api.Harness, "status", boom)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    code, _ = run_cli("--workspace", str(workspace), "status")
    assert code == 1
    assert "hunter2" not in err.getvalue()
    assert "RuntimeError" in err.getvalue()


def test_cli_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    from crystallizer import api

    def interrupt(self: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(api.Harness, "status", interrupt)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert run_cli("--workspace", str(workspace), "status")[0] == 1


def test_mock_script_missing(workspace: Path) -> None:
    (workspace / "crystallizer.toml").write_text(
        '[model]\nmock_script = "nope.json"\n', encoding="utf-8"
    )
    code, _ = run_cli("--workspace", str(workspace), "plan", "goal")
    assert code == 2
