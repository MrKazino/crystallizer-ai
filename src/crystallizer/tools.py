"""Sandboxed tools: file_read, file_write, file_delete, file_search, file_replace, shell, run_tests.

The sandbox confines paths, executables, environment and time. It is **not** OS-level isolation:
code run by tests or allowed executables can do anything the user can. Run untrusted work in a
container.

Rules enforced here:

* every path resolves inside the workspace; ``..`` components and symlink escapes are rejected;
* nothing inside ``state_dir`` can be read, written or deleted by a tool;
* writes and deletes inside protected paths (default ``.git``) are rejected, which blocks
  hook injection;
* writes open the final component with ``O_NOFOLLOW``;
* ``shell`` takes an argv list, never a shell string; the executable must be allow-listed by
  bare name; ``python`` resolves to the interpreter running crystallizer;
* subprocesses get only allow-listed environment variables, no stdin, and a timeout;
* output is redacted, truncated to ``max_output_bytes`` and always treated as untrusted data.
"""

from __future__ import annotations

import os
import posixpath
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from crystallizer.config import ToolsConfig
from crystallizer.errors import ExtensionError, SandboxError, ToolError
from crystallizer.redaction import Redactor
from crystallizer.schemas import Action, ArgValue, ToolResult, ToolSpec

TRUNCATION_MARKER = "\n[truncated]"
PATH_ARGS: dict[str, tuple[str, ...]] = {
    "file_read": ("path",),
    "file_write": ("path",),
    "file_delete": ("path",),
}


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FileReadArgs(_Args):
    """Arguments of ``file_read``."""

    path: str = Field(min_length=1, description="Workspace-relative file path.")


class FileWriteArgs(_Args):
    """Arguments of ``file_write``."""

    path: str = Field(min_length=1, description="Workspace-relative file path.")
    content: str = Field(description="Full new file content (UTF-8).")


class FileDeleteArgs(_Args):
    """Arguments of ``file_delete``."""

    path: str = Field(min_length=1, description="Workspace-relative file path.")


class FileSearchArgs(_Args):
    """Arguments of ``file_search``."""

    query: str = Field(min_length=1, description="Literal text to find.")
    glob: str = Field(default="**/*", description="Workspace-relative glob of files to search.")


class FileReplaceArgs(_Args):
    """Arguments of ``file_replace``."""

    glob: str = Field(min_length=1, description="Workspace-relative glob of files to edit.")
    old: str = Field(min_length=1, description="Literal text to replace.")
    new: str = Field(description="Replacement text.")
    word: bool = Field(default=True, description="Only replace whole identifiers.")


class ShellArgs(_Args):
    """Arguments of ``shell``."""

    argv: list[str] = Field(min_length=1, description="Command as an argv list (no shell).")


class RunTestsArgs(_Args):
    """Arguments of ``run_tests``."""

    args: list[str] = Field(default_factory=list, description="Extra pytest arguments.")


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate ``text`` to at most ``limit`` UTF-8 bytes (plus a marker)."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, False
    return raw[:limit].decode("utf-8", errors="ignore") + TRUNCATION_MARKER, True


def normalize_action(action: Action) -> Action:
    """Return ``action`` with path arguments normalized (``./src/x.py`` -> ``src/x.py``)."""
    names = PATH_ARGS.get(action.tool, ())
    if not names:
        return action
    args = dict(action.args)
    for name in names:
        value = args.get(name)
        if isinstance(value, str) and value:
            args[name] = posixpath.normpath(value)
    return Action(tool=action.tool, args=args)


class Sandbox:
    """Path, executable, environment and time confinement for tool calls."""

    def __init__(
        self,
        workspace: Path,
        state_dir: Path,
        config: ToolsConfig,
        redactor: Redactor,
        environ: Mapping[str, str] | None = None,
        python_executable: str | None = None,
    ) -> None:
        """Create a sandbox rooted at ``workspace``."""
        self.workspace = workspace.resolve()
        self.state_dir = state_dir.resolve()
        self.config = config
        self.redactor = redactor
        self._environ = dict(os.environ if environ is None else environ)
        self._python = python_executable or sys.executable
        self._protected = [(self.workspace / item).resolve() for item in config.protected_paths]

    # ------------------------------------------------------------------ paths

    def resolve(self, raw: str, *, write: bool) -> Path:
        """Resolve ``raw`` to an absolute path inside the workspace or raise SandboxError."""
        if not raw or "\x00" in raw:
            raise SandboxError("empty or invalid path")
        candidate = Path(raw)
        if ".." in candidate.parts:
            raise SandboxError(f"path traversal is not allowed: {raw}")
        full = candidate if candidate.is_absolute() else self.workspace / candidate
        resolved = full.resolve()
        if resolved == self.workspace or not resolved.is_relative_to(self.workspace):
            raise SandboxError(f"path escapes the workspace: {raw}")
        if resolved == self.state_dir or resolved.is_relative_to(self.state_dir):
            raise SandboxError(f"path is inside the protected state directory: {raw}")
        if write and self._is_protected(resolved):
            raise SandboxError(f"path is protected: {raw}")
        return resolved

    def _is_protected(self, path: Path) -> bool:
        return any(path == item or path.is_relative_to(item) for item in self._protected)

    def relative(self, path: Path) -> str:
        """Workspace-relative POSIX form of ``path``."""
        return path.relative_to(self.workspace).as_posix()

    def iter_files(self, pattern: str, *, write: bool) -> Iterator[Path]:
        """Yield files matching ``pattern`` that the sandbox allows, in sorted order."""
        if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise SandboxError(f"invalid glob: {pattern}")
        for match in sorted(self.workspace.glob(pattern)):
            try:
                resolved = self.resolve(self.relative(match.absolute()), write=write)
            except (SandboxError, ValueError):
                continue
            if resolved.is_file() and resolved.stat().st_size <= self.config.max_file_bytes:
                yield resolved

    # ------------------------------------------------------------------ results

    def _result(self, ok: bool, text: str, exit_code: int | None = None) -> ToolResult:
        clean = self.redactor.redact(text)
        output, cut = truncate(clean, self.config.max_output_bytes)
        return ToolResult(ok=ok, exit_code=exit_code, output=output, truncated=cut)

    # ------------------------------------------------------------------ file tools

    def file_read(self, args: FileReadArgs) -> ToolResult:
        """Read a UTF-8 file (capped at ``max_file_bytes``)."""
        path = self.resolve(args.path, write=False)
        if not path.is_file():
            return self._result(False, f"not a file: {args.path}")
        with path.open("rb") as handle:
            raw = handle.read(self.config.max_file_bytes + 1)
        text = raw[: self.config.max_file_bytes].decode("utf-8", errors="replace")
        if len(raw) > self.config.max_file_bytes:
            text += TRUNCATION_MARKER
        return self._result(True, text)

    def file_write(self, args: FileWriteArgs) -> ToolResult:
        """Create or overwrite a file inside the workspace."""
        data = args.content.encode("utf-8")
        if len(data) > self.config.max_file_bytes:
            return self._result(False, f"content exceeds max_file_bytes for {args.path}")
        path = self.resolve(args.path, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.parent.resolve().is_relative_to(self.workspace):
            raise SandboxError(f"path escapes the workspace: {args.path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return self._result(True, f"wrote {len(data)} bytes to {self.relative(path)}")

    def file_delete(self, args: FileDeleteArgs) -> ToolResult:
        """Delete one file inside the workspace."""
        path = self.resolve(args.path, write=True)
        if not path.is_file():
            return self._result(False, f"not a file: {args.path}")
        path.unlink()
        return self._result(True, f"deleted {self.relative(path)}")

    def file_search(self, args: FileSearchArgs) -> ToolResult:
        """List ``path:line: text`` for every line containing ``query``."""
        lines: list[str] = []
        for path in self.iter_files(args.glob, write=False):
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if args.query in line:
                    lines.append(f"{self.relative(path)}:{number}: {line.strip()}")
        return self._result(True, "\n".join(lines) if lines else "no matches")

    def file_replace(self, args: FileReplaceArgs) -> ToolResult:
        """Replace ``old`` with ``new`` in every file matching ``glob``."""
        if args.word:
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(args.old)}(?![A-Za-z0-9_])")
        else:
            pattern = re.compile(re.escape(args.old))
        report: list[str] = []
        total = 0
        for path in self.iter_files(args.glob, write=True):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            updated, count = pattern.subn(lambda _m: args.new, text)
            if count:
                encoded = updated.encode("utf-8")
                if len(encoded) > self.config.max_file_bytes:
                    return self._result(False, f"result exceeds max_file_bytes: {path.name}")
                self.file_write(FileWriteArgs(path=self.relative(path), content=updated))
                report.append(f"{self.relative(path)}: {count}")
                total += count
        report.append(f"total replacements: {total}")
        return self._result(True, "\n".join(report))

    # ------------------------------------------------------------------ processes

    def environment(self) -> dict[str, str]:
        """The environment passed to subprocesses: allow-listed variables only.

        ``PATH`` keeps only absolute entries outside the workspace, so an agent-written file in the
        workspace can never shadow an allowed executable (e.g. a ``./git`` with ``.`` on PATH).
        """
        env = {
            name: self._environ[name] for name in self.config.env_allowlist if name in self._environ
        }
        if "PATH" in env:
            env["PATH"] = self._safe_path(env["PATH"])
        return env

    def _safe_path(self, value: str) -> str:
        kept: list[str] = []
        for entry in value.split(os.pathsep):
            if not entry or not os.path.isabs(entry):
                continue
            resolved = Path(entry).resolve()
            if resolved == self.workspace or resolved.is_relative_to(self.workspace):
                continue
            kept.append(entry)
        return os.pathsep.join(kept)

    def shell(self, args: ShellArgs) -> ToolResult:
        """Run an allow-listed executable with an argv list, a timeout, and a filtered env."""
        executable = args.argv[0]
        if executable not in self.config.allowed_executables:
            raise SandboxError(f"executable is not allowed: {executable}")
        env = self.environment()
        if executable == "python":
            program: str | None = self._python
        else:
            program = shutil.which(executable, path=env.get("PATH"))
        if program is None:
            return self._result(False, f"executable not found: {executable}")
        try:
            completed = subprocess.run(
                [program, *args.argv[1:]],
                cwd=self.workspace,
                env=env,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _decode(exc.stdout) + _decode(exc.stderr)
            return self._result(
                False, f"timed out after {self.config.timeout_seconds}s\n{partial}", None
            )
        except OSError as exc:
            return self._result(False, f"could not start {executable}: {exc}")
        output = _decode(completed.stdout)
        errors = _decode(completed.stderr)
        if errors:
            output = f"{output}\n[stderr]\n{errors}" if output else f"[stderr]\n{errors}"
        return self._result(completed.returncode == 0, output, completed.returncode)

    def run_tests(self, args: RunTestsArgs) -> ToolResult:
        """Run ``python -m pytest`` with extra arguments."""
        return self.shell(ShellArgs(argv=["python", "-m", "pytest", *args.args]))


def _decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


ToolHandler = Callable[[Mapping[str, ArgValue], Sandbox], ToolResult]

_BUILTINS: tuple[tuple[str, str, type[_Args], bool], ...] = (
    ("file_read", "Read a UTF-8 text file inside the workspace.", FileReadArgs, True),
    ("file_write", "Create or overwrite a text file inside the workspace.", FileWriteArgs, True),
    ("file_delete", "Delete a file inside the workspace (irreversible).", FileDeleteArgs, False),
    ("file_search", "Find lines containing literal text in workspace files.", FileSearchArgs, True),
    ("file_replace", "Replace literal text in workspace files.", FileReplaceArgs, True),
    ("shell", "Run an allow-listed executable with an argv list.", ShellArgs, True),
    ("run_tests", "Run pytest in the workspace.", RunTestsArgs, True),
)


def _builtin_handler(name: str, model: type[_Args]) -> ToolHandler:
    def handler(args: Mapping[str, ArgValue], sandbox: Sandbox) -> ToolResult:
        parsed = model.model_validate(dict(args))
        method: Callable[[Any], ToolResult] = getattr(sandbox, name)
        return method(parsed)

    return handler


class ToolRegistry:
    """Built-in and plugin tools, keyed by name. Built-in names cannot be replaced."""

    def __init__(self, sandbox: Sandbox) -> None:
        """Register the built-in tools."""
        self.sandbox = sandbox
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}
        for name, description, model, reversible in _BUILTINS:
            spec = ToolSpec(
                name=name,
                description=description,
                args_schema=model.model_json_schema(),
                reversible=reversible,
                builtin=True,
            )
            self._tools[name] = (spec, _builtin_handler(name, model))

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Add a plugin tool."""
        if spec.name in self._tools:
            raise ExtensionError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = (spec.model_copy(update={"builtin": False}), handler)

    def spec(self, name: str) -> ToolSpec | None:
        """Return the descriptor of ``name`` or None."""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def specs(self) -> list[ToolSpec]:
        """All descriptors, sorted by name."""
        return [self._tools[name][0] for name in sorted(self._tools)]

    def execute(self, action: Action) -> ToolResult:
        """Run ``action``. Sandbox and argument violations become failed results."""
        entry = self._tools.get(action.tool)
        if entry is None:
            return ToolResult(ok=False, output=f"unknown tool: {action.tool}")
        _, handler = entry
        try:
            return handler(action.args, self.sandbox)
        except ValidationError as exc:
            message = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            return ToolResult(ok=False, output=f"invalid arguments for {action.tool}: {message}")
        except (SandboxError, ToolError) as exc:
            return ToolResult(ok=False, output=exc.message)
