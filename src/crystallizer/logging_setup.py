"""Structured JSON-lines logging with redaction of every message and every extra field.

Each line has ``ts``, ``level``, ``module``, ``message``, ``run_id``, ``task_id`` plus any extra
fields passed with ``logger.info(..., extra={...})``. ``run_id``/``task_id`` come from a context
set with :func:`log_context`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO, Any

from crystallizer.clock import Clock, iso
from crystallizer.redaction import Redactor, default_redactor

LOGGER_NAME = "crystallizer"

_run_id: ContextVar[str | None] = ContextVar("crystallizer_run_id", default=None)
_task_id: ContextVar[str | None] = ContextVar("crystallizer_task_id", default=None)

_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None)).keys()
) | {"message", "asctime", "taskName"}


@contextmanager
def log_context(run_id: str | None = None, task_id: str | None = None) -> Iterator[None]:
    """Attach ``run_id`` and ``task_id`` to every log line emitted inside the block."""
    run_token = _run_id.set(run_id if run_id is not None else _run_id.get())
    task_token = _task_id.set(task_id)
    try:
        yield
    finally:
        _task_id.reset(task_token)
        _run_id.reset(run_token)


class JsonFormatter(logging.Formatter):
    """Formats records as redacted JSON lines."""

    def __init__(self, clock: Clock | None = None, redactor: Redactor | None = None) -> None:
        """Use ``clock`` for timestamps when given, else the record creation time."""
        super().__init__()
        self._clock = clock
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as one JSON line."""
        redactor = self._redactor or default_redactor()
        if self._clock is not None:
            ts = iso(self._clock.now())
        else:
            ts = iso(datetime.fromtimestamp(record.created, UTC))
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "module": record.module,
            "message": redactor.redact(record.getMessage()),
            "run_id": _run_id.get(),
            "task_id": _task_id.get(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and key not in payload:
                payload[key] = redactor.redact_obj(_plain(value))
        if record.exc_info and record.exc_info[1] is not None:
            payload["error"] = redactor.redact(repr(record.exc_info[1]))
        return json.dumps(payload, sort_keys=True, default=str)


def _plain(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(v) for v in value]
    return str(value)


def setup_logging(
    *,
    verbose: bool = False,
    debug: bool = False,
    stream: IO[str] | None = None,
    clock: Clock | None = None,
    redactor: Redactor | None = None,
) -> logging.Logger:
    """Configure the ``crystallizer`` logger. ``--verbose`` = INFO, ``--debug`` = DEBUG."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(clock=clock, redactor=redactor))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``crystallizer`` logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
