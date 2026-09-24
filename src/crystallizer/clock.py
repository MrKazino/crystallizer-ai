"""Injected time and randomness.

Logic never calls ``time.time()``, ``datetime.now()`` or the global ``random`` module directly.
It receives a :class:`Clock` and a seeded :class:`random.Random`, so tests are deterministic.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Source of the current time (always timezone-aware UTC)."""

    def now(self) -> datetime:
        """Return the current time."""
        ...


class SystemClock:
    """Wall-clock time. Used only by the CLI entry point."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(UTC)


class FixedClock:
    """Deterministic clock that starts at ``start`` and advances by ``step`` on every read."""

    def __init__(self, start: datetime | None = None, step: timedelta | None = None) -> None:
        """Create a clock; defaults to 2026-01-01T00:00:00Z advancing one second per read."""
        self._current = start or datetime(2026, 1, 1, tzinfo=UTC)
        if self._current.tzinfo is None:
            self._current = self._current.replace(tzinfo=UTC)
        self._step = timedelta(seconds=1) if step is None else step

    def now(self) -> datetime:
        """Return the current time, then advance by the step."""
        value = self._current
        self._current = self._current + self._step
        return value

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward by ``delta``."""
        self._current = self._current + delta


def make_rng(seed: int) -> random.Random:
    """Return an isolated, seeded random generator."""
    return random.Random(seed)  # noqa: S311 - not used for security


def iso(moment: datetime) -> str:
    """Format a datetime as an ISO-8601 UTC string with a trailing ``Z``."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(text: str) -> datetime:
    """Parse a string produced by :func:`iso`."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def new_run_id(clock: Clock, rng: random.Random) -> str:
    """Return a sortable run identifier such as ``run-20260101T000000-a1b2``."""
    stamp = clock.now().astimezone(UTC).strftime("%Y%m%dT%H%M%S")
    return f"run-{stamp}-{rng.getrandbits(16):04x}"
