"""Synchronous event bus for observers (logging, metrics, plugins, external workflows).

Events are immutable and redacted before delivery. Observers cannot change routing or execution:
nothing the harness decides reads events back. An observer that raises is logged and counted,
never propagated, so a faulty plugin cannot break a run.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import JsonValue

from crystallizer.clock import Clock, iso
from crystallizer.hashing import to_jsonable
from crystallizer.logging_setup import get_logger
from crystallizer.redaction import Redactor
from crystallizer.schemas import Event

EventHandler = Callable[[Event], None]
_log = get_logger("events")


class EventKind(StrEnum):
    """Kinds of events the harness publishes."""

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    TASK_STARTED = "task.started"
    TASK_FINISHED = "task.finished"
    STEP_ROUTED = "step.routed"
    STEP_EXECUTED = "step.executed"
    ESCALATION = "escalation"
    SKILL_MINED = "skill.mined"
    SKILL_PROMOTED = "skill.promoted"
    SKILL_DEMOTED = "skill.demoted"
    CHECKPOINT_SAVED = "checkpoint.saved"
    BUDGET_EXHAUSTED = "budget.exhausted"


class EventBus:
    """Publish/subscribe with isolation of observer failures."""

    def __init__(self, clock: Clock, redactor: Redactor) -> None:
        """Create an empty bus."""
        self._clock = clock
        self._redactor = redactor
        self._handlers: list[EventHandler] = []
        self.errors = 0

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Add ``handler``; returns a function that unsubscribes it."""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    def publish(
        self,
        kind: EventKind,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        **data: Any,
    ) -> Event:
        """Build a redacted event and deliver it to every handler."""
        clean: dict[str, JsonValue] = self._redactor.redact_obj(to_jsonable(data))
        event = Event(
            kind=kind.value,
            ts=iso(self._clock.now()),
            run_id=run_id,
            task_id=task_id,
            data=clean,
        )
        for handler in list(self._handlers):
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - observers must never break a run
                self.errors += 1
                _log.warning(
                    "event handler failed", extra={"event": event.kind, "error": repr(exc)}
                )
        return event
