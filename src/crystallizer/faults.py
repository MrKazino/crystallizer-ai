"""Deterministic fault injection used by crash-and-resume tests.

Production code calls :meth:`FaultInjector.hit` at named points. The default injector never
fires. Tests configure one point and a hit count to raise :class:`SimulatedCrash` exactly there.
"""

from __future__ import annotations

from crystallizer.errors import SimulatedCrash

FAULT_POINTS = frozenset(
    {
        "journal.after_begin",
        "journal.after_execute",
        "runner.after_step",
        "runner.before_checkpoint",
        "runner.after_acceptance",
    }
)


class FaultInjector:
    """Raise :class:`SimulatedCrash` on the ``after``-th hit of ``point``."""

    def __init__(self, point: str | None = None, after: int = 1) -> None:
        """Configure the crash point; ``point=None`` disables injection."""
        if point is not None and point not in FAULT_POINTS:
            raise ValueError(f"unknown fault point: {point}")
        if after < 1:
            raise ValueError("after must be >= 1")
        self.point = point
        self.after = after
        self.hits = 0

    def hit(self, point: str) -> None:
        """Record a pass through ``point`` and crash if it is the configured one."""
        if point != self.point:
            return
        self.hits += 1
        if self.hits == self.after:
            raise SimulatedCrash(f"simulated crash at {point} (hit {self.hits})")


NO_FAULTS = FaultInjector()
