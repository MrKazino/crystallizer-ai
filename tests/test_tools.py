"""Sandbox tools: confinement, protected paths, subprocess rules, output handling."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from crystallizer.config import ToolsConfig
from crystallizer.errors import ExtensionError, SandboxError
from crystallizer.redaction import REDACTED, Redactor
from crystallizer.schemas import Action, ArgValue, ToolResult, ToolSpec
from crystallizer.tools import (
    FileReadArgs,
    FileWriteArgs,
    Sandbox,
    ShellArgs,
    ToolRegistry,
    normalize_action,
    truncate,
)


def run(sandbox: Sandbox, tool: str, **args: ArgValue) -> ToolResult:
    return ToolRegistry(sandbox).execute(Action(tool=tool, args=args))


def test_write_read_delete(sandbox: Sandbox, workspace: Path) -> None:
    assert run(sandbox, "file_write", path="src/pkg/a.py", content="x = 1\n").ok
    assert (workspace / "src/pkg/a.py").read_text(encoding="utf-8") == "x = 1\n"
    result = run(sandbox, "file_read", path="./src/pkg/a.py")
    assert result.ok
    assert result.output == "x = 1\n"
    assert run(sandbox, "file_delete", path="src/pkg/a.py").ok
    assert not run(sandbox, "file_read", path="src/pkg/a.py").ok
    assert not run(sandbox, "file_delete", path="src/pkg/a.py").ok


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../x", "/etc/passwd", "", "a\x00b"])
def test_path_traversal_blocked(sandbox: Sandbox, path: str) -> None:
    result = run(sandbox, "file_write", path=path, content="x")
    assert not result.ok
    with pytest.raises(SandboxError):
        sandbox.resolve(path, write=True)


def test_workspace_root_itself_is_rejected(sandbox: Sandbox, workspace: Path) -> None:
    with pytest.raises(SandboxError):
        sandbox.resolve(str(workspace), write=False)


def test_absolute_path_inside_workspace_allowed(sandbox: Sandbox, workspace: Path) -> None:
    assert sandbox.resolve(str(workspace / "a.txt"), write=True) == workspace.resolve() / "a.txt"


def test_symlink_escape_blocked(sandbox: Sandbox, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    assert not run(sandbox, "file_read", path="link/secret.txt").ok
    assert not run(sandbox, "file_write", path="link/new.txt", content="x").ok
    assert not (outside / "new.txt").exists()
    (workspace / "file_link").symlink_to(outside / "secret.txt")
    assert not run(sandbox, "file_write", path="file_link", content="x").ok
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "s"


def test_symlink_inside_workspace_allowed(sandbox: Sandbox, workspace: Path) -> None:
    (workspace / "real.txt").write_text("r", encoding="utf-8")
    (workspace / "alias.txt").symlink_to(workspace / "real.txt")
    assert run(sandbox, "file_read", path="alias.txt").output == "r"


def test_state_dir_is_unreachable(sandbox: Sandbox, state_dir: Path) -> None:
    (state_dir / "manifest.json").write_text("{}", encoding="utf-8")
    cases: list[tuple[str, dict[str, ArgValue]]] = [
        ("file_write", {"path": ".crystallizer/manifest.json", "content": "x"}),
        ("file_delete", {"path": ".crystallizer/manifest.json"}),
        ("file_read", {"path": ".crystallizer/manifest.json"}),
        ("file_write", {"path": ".crystallizer/skills/active/evil.json", "content": "x"}),
    ]
    for tool, args in cases:
        assert not ToolRegistry(sandbox).execute(Action(tool=tool, args=args)).ok
    assert (state_dir / "manifest.json").read_text(encoding="utf-8") == "{}"
    replace = run(sandbox, "file_replace", glob="**/*.json", old="{}", new="[]", word=False)
    assert replace.ok
    assert (state_dir / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_protected_git_dir(sandbox: Sandbox, workspace: Path) -> None:
    (workspace / ".git" / "hooks").mkdir(parents=True)
    assert not run(sandbox, "file_write", path=".git/hooks/pre-commit", content="#!/bin/sh").ok
    assert not (workspace / ".git/hooks/pre-commit").exists()
    (workspace / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    assert run(sandbox, "file_read", path=".git/HEAD").ok


def test_file_size_limits(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(workspace, state_dir, ToolsConfig(max_file_bytes=10), Redactor())
    assert not run(sandbox, "file_write", path="a.txt", content="x" * 11).ok
    (workspace / "big.txt").write_text("y" * 20, encoding="utf-8")
    result = sandbox.file_read(FileReadArgs(path="big.txt"))
    assert result.ok
    assert result.output.endswith("[truncated]")


def test_invalid_arguments_and_unknown_tool(sandbox: Sandbox) -> None:
    assert "invalid arguments" in run(sandbox, "file_write", path="a.txt").output
    assert "invalid arguments" in run(sandbox, "file_write", path="a", content=3).output
    assert run(sandbox, "teleport").output == "unknown tool: teleport"


def test_file_search_and_replace(sandbox: Sandbox, workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src/a.py").write_text("def total():\n    return subtotal()\n", encoding="utf-8")
    (workspace / "src/b.py").write_text("from a import total\n", encoding="utf-8")
    (workspace / "src/bin.dat").write_bytes(b"\xff\xfe total")
    found = run(sandbox, "file_search", query="total", glob="src/*.py")
    assert found.output.splitlines() == [
        "src/a.py:1: def total():",
        "src/a.py:2: return subtotal()",
        "src/b.py:1: from a import total",
    ]
    assert run(sandbox, "file_search", query="nothing").output == "no matches"
    replaced = run(sandbox, "file_replace", glob="src/*", old="total", new="grand_total")
    assert replaced.ok
    assert "total replacements: 2" in replaced.output
    assert "subtotal()" in (workspace / "src/a.py").read_text(encoding="utf-8")
    assert "grand_total" in (workspace / "src/b.py").read_text(encoding="utf-8")
    loose = run(sandbox, "file_replace", glob="src/a.py", old="sub", new="", word=False)
    assert "total replacements: 1" in loose.output
    assert not run(sandbox, "file_search", query="x", glob="../*").ok
    assert not run(sandbox, "file_search", query="x", glob="/etc/*").ok


def test_file_replace_respects_size_limit(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(workspace, state_dir, ToolsConfig(max_file_bytes=12), Redactor())
    (workspace / "a.txt").write_text("x x x", encoding="utf-8")
    result = run(sandbox, "file_replace", glob="*.txt", old="x", new="longer")
    assert not result.ok


def test_shell_allowlist_and_env(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(
        workspace,
        state_dir,
        ToolsConfig(),
        Redactor(),
        environ={"PATH": "/usr/bin:/bin", "SECRET_TOKEN": "zzz", "HOME": "/tmp"},
    )
    assert sandbox.environment() == {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    with pytest.raises(SandboxError):
        sandbox.shell(ShellArgs(argv=["rm", "-rf", "/"]))
    assert not run(sandbox, "shell", argv=["/bin/sh", "-c", "echo hi"]).ok
    code = "import os,sys; print(sorted(os.environ)); print('password=hunter2', file=sys.stderr)"
    result = run(sandbox, "shell", argv=["python", "-c", code])
    assert result.ok
    assert result.exit_code == 0
    assert "SECRET_TOKEN" not in result.output
    assert f"password={REDACTED}" in result.output
    assert "hunter2" not in result.output
    failing = run(sandbox, "shell", argv=["python", "-c", "raise SystemExit(3)"])
    assert not failing.ok
    assert failing.exit_code == 3


def test_shell_timeout(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(workspace, state_dir, ToolsConfig(timeout_seconds=1), Redactor())
    result = run(sandbox, "shell", argv=["python", "-c", "import time; time.sleep(5)"])
    assert not result.ok
    assert result.exit_code is None
    assert "timed out" in result.output


def test_shell_missing_executable(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(
        workspace,
        state_dir,
        ToolsConfig(allowed_executables=["definitely-missing-tool"]),
        Redactor(),
        environ={"PATH": "/nonexistent"},
    )
    assert "not found" in run(sandbox, "shell", argv=["definitely-missing-tool"]).output


def test_shell_start_failure(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(
        workspace, state_dir, ToolsConfig(), Redactor(), python_executable=str(workspace / "nope")
    )
    assert "could not start" in run(sandbox, "shell", argv=["python", "-V"]).output


def test_output_truncation(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(workspace, state_dir, ToolsConfig(max_output_bytes=256), Redactor())
    result = run(sandbox, "shell", argv=["python", "-c", "print('a' * 5000)"])
    assert result.truncated
    assert result.output.endswith("[truncated]")
    assert truncate("héllo", 2) == ("h\n[truncated]", True)


def test_run_tests(workspace: Path, state_dir: Path) -> None:
    sandbox = Sandbox(
        workspace, state_dir, ToolsConfig(), Redactor(), python_executable=sys.executable
    )
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = run(sandbox, "run_tests", args=["-q", "-p", "no:cacheprovider", "test_ok.py"])
    assert result.ok, result.output
    assert "1 passed" in result.output


def test_normalize_action() -> None:
    action = Action(tool="file_write", args={"path": "./src//a.py", "content": "x"})
    assert normalize_action(action).args["path"] == "src/a.py"
    shell = Action(tool="shell", args={"argv": ["git", "status"]})
    assert normalize_action(shell) is shell
    empty = Action(tool="file_read", args={"path": ""})
    assert normalize_action(empty).args["path"] == ""


def test_plugin_tool_registration(sandbox: Sandbox) -> None:
    registry = ToolRegistry(sandbox)

    def echo(args: Mapping[str, ArgValue], box: Sandbox) -> ToolResult:
        return ToolResult(ok=True, output=str(args.get("text", "")))

    spec = ToolSpec(name="echo", description="Echo text.", args_schema={}, reversible=True)
    registry.register(spec, echo)
    assert registry.execute(Action(tool="echo", args={"text": "hi"})).output == "hi"
    stored = registry.spec("echo")
    assert stored is not None
    assert not stored.builtin
    assert registry.spec("missing") is None
    assert [s.name for s in registry.specs()][:2] == ["echo", "file_delete"]
    with pytest.raises(ExtensionError):
        registry.register(spec.model_copy(update={"name": "shell"}), echo)


def test_tool_specs_export_json_schema(sandbox: Sandbox) -> None:
    specs = {spec.name: spec for spec in ToolRegistry(sandbox).specs()}
    assert set(specs) == {
        "file_delete",
        "file_read",
        "file_replace",
        "file_search",
        "file_write",
        "run_tests",
        "shell",
    }
    assert specs["shell"].args_schema["required"] == ["argv"]
    assert not specs["file_delete"].reversible


def test_write_uses_nofollow(sandbox: Sandbox, workspace: Path) -> None:
    result = sandbox.file_write(FileWriteArgs(path="plain.txt", content="ok"))
    assert result.ok
    assert (workspace / "plain.txt").read_text(encoding="utf-8") == "ok"


def test_path_entries_inside_workspace_are_dropped(workspace: Path, state_dir: Path) -> None:
    (workspace / "bin").mkdir()
    fake_git = workspace / "bin" / "git"
    fake_git.write_text("#!/bin/sh\necho hijacked\n", encoding="utf-8")
    fake_git.chmod(0o755)
    path_value = os.pathsep.join([".", str(workspace / "bin"), "relative/dir", "/usr/bin", "/bin"])
    sandbox = Sandbox(workspace, state_dir, ToolsConfig(), Redactor(), environ={"PATH": path_value})
    assert sandbox.environment()["PATH"] == os.pathsep.join(["/usr/bin", "/bin"])
    result = run(sandbox, "shell", argv=["git", "--version"])
    assert "hijacked" not in result.output
