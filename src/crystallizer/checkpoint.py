"""Atomic, verified checkpoints of project state.

Each checkpoint is written to a temporary file, flushed, fsynced, and atomically renamed; the
directory is fsynced afterwards. Its ``hash`` is the SHA-256 of the canonical JSON of every other
field. The newest three are kept. Loading walks from newest to oldest and returns the first one
whose hash verifies; if checkpoint files exist but none verifies, :class:`CheckpointError`
(exit code 6) is raised.
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

from pydantic import ValidationError

from crystallizer.clock import Clock, iso
from crystallizer.errors import CheckpointError
from crystallizer.hashing import canonical_json, digest
from crystallizer.logging_setup import get_logger
from crystallizer.redaction import Redactor
from crystallizer.schemas import Checkpoint, Cursor, Plan

KEEP = 3
_NAME = re.compile(r"^ckpt-(\d{8})\.json$")
_log = get_logger("checkpoint")


def checkpoint_hash(checkpoint: Checkpoint) -> str:
    """SHA-256 over the canonical JSON of the checkpoint without its ``hash`` field."""
    data = checkpoint.model_dump(mode="json")
    data.pop("hash", None)
    return digest(data)


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file, fsync, rename, fsync directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class CheckpointStore:
    """Checkpoint files under ``state_dir/checkpoints``."""

    def __init__(self, state_dir: Path, clock: Clock, redactor: Redactor, keep: int = KEEP) -> None:
        """Bind to ``state_dir``."""
        self.directory = state_dir / "checkpoints"
        self._clock = clock
        self._redactor = redactor
        self._keep = keep

    def _files(self) -> list[tuple[int, Path]]:
        if not self.directory.is_dir():
            return []
        found = []
        for path in self.directory.iterdir():
            match = _NAME.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return sorted(found)

    def save(
        self, plan: Plan, cursor: Cursor, memory_snapshot_id: int, registry_version: int
    ) -> Checkpoint:
        """Write a new checkpoint and prune old ones."""
        files = self._files()
        seq = files[-1][0] + 1 if files else 1
        clean_plan = Plan.model_validate(self._redactor.redact_obj(plan.model_dump(mode="json")))
        checkpoint = Checkpoint(
            seq=seq,
            created_at=iso(self._clock.now()),
            plan=clean_plan,
            memory_snapshot_id=memory_snapshot_id,
            registry_version=registry_version,
            cursor=cursor,
        )
        checkpoint = checkpoint.model_copy(update={"hash": checkpoint_hash(checkpoint)})
        atomic_write(self.directory / f"ckpt-{seq:08d}.json", canonical_json(checkpoint) + "\n")
        for _, old in self._files()[: -self._keep]:
            with contextlib.suppress(FileNotFoundError):
                old.unlink()
        return checkpoint

    def load(self, path: Path) -> Checkpoint:
        """Load and verify one checkpoint file."""
        try:
            checkpoint = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise CheckpointError(
                f"unreadable checkpoint {path.name}: {type(exc).__name__}"
            ) from None
        if checkpoint.hash != checkpoint_hash(checkpoint):
            raise CheckpointError(f"checkpoint {path.name} failed hash verification")
        return checkpoint

    def latest(self) -> Checkpoint | None:
        """Return the newest valid checkpoint, falling back past corrupt ones."""
        files = self._files()
        if not files:
            return None
        for _, path in reversed(files):
            try:
                return self.load(path)
            except CheckpointError as exc:
                _log.warning("skipping invalid checkpoint", extra={"reason": exc.message})
        raise CheckpointError("no valid checkpoint could be restored")

    def exists(self) -> bool:
        """True if any checkpoint file exists."""
        return bool(self._files())
