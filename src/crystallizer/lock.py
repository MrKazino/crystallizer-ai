"""Exclusive per-workspace run lock.

``state_dir/run.lock`` is created with ``O_CREAT | O_EXCL`` and holds the owner's PID. A lock
whose PID is no longer alive is stale: it is removed and acquisition retried once. A live lock
fails with :class:`WorkspaceLockedError` (exit code 5). The lock is always released in ``finally``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from crystallizer.errors import WorkspaceLockedError
from crystallizer.logging_setup import get_logger

LOCK_NAME = "run.lock"
_log = get_logger("lock")


def pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """Context manager holding the workspace run lock."""

    def __init__(
        self,
        state_dir: Path,
        pid: int | None = None,
        is_alive: Callable[[int], bool] = pid_alive,
    ) -> None:
        """Prepare a lock in ``state_dir`` owned by ``pid`` (default: this process)."""
        self.path = state_dir / LOCK_NAME
        self.pid = os.getpid() if pid is None else pid
        self._is_alive = is_alive
        self.held = False

    def acquire(self) -> None:
        """Take the lock or raise :class:`WorkspaceLockedError`."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner = self._read_owner()
                if owner is not None and self._is_alive(owner):
                    raise WorkspaceLockedError(
                        f"workspace is locked by running process {owner} ({self.path})"
                    ) from None
                _log.warning("removing stale lock", extra={"stale_pid": owner})
                self._remove_stale()
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(self.pid))
            self.held = True
            return
        raise WorkspaceLockedError(f"could not acquire workspace lock ({self.path})")

    def _remove_stale(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def _read_owner(self) -> int | None:
        try:
            text = self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return int(text) if text.isdigit() else None

    def release(self) -> None:
        """Release the lock if this instance holds it."""
        if not self.held:
            return
        self.held = False
        if self._read_owner() == self.pid:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()

    def __enter__(self) -> RunLock:
        """Acquire on entry."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Always release."""
        self.release()
