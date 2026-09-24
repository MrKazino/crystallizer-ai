"""Append-only JSONL traces: one file per run in ``state_dir/traces/``.

Three record types share a file: ``step`` (:class:`TraceStep`), ``verdict``
(:class:`TraceVerdict`) and ``overhead`` (:class:`TraceOverhead`, usage that executed nothing).
Records are redacted before they are written, flushed and fsynced per line. A step whose action or
situation was changed by redaction is flagged ``redacted`` and is never mined. Loading derives
``verified`` from the latest verdict for each task attempt and skips a torn final line left by a
crash.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crystallizer.logging_setup import get_logger
from crystallizer.models import CostTable
from crystallizer.redaction import Redactor
from crystallizer.schemas import (
    Action,
    CostReport,
    RouteCost,
    TraceOverhead,
    TraceStep,
    TraceVerdict,
    Usage,
)
from crystallizer.tools import normalize_action

_log = get_logger("traces")


class TraceRecorder:
    """Appends redacted records for one run."""

    def __init__(self, directory: Path, run_id: str, redactor: Redactor) -> None:
        """Write to ``directory/<run_id>.jsonl``."""
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{run_id}.jsonl"
        self._redactor = redactor

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_step(self, step: TraceStep) -> TraceStep:
        """Redact, flag, and append a step; return the stored form."""
        raw = step.model_dump(mode="json")
        clean: dict[str, Any] = self._redactor.redact_obj(raw)
        changed = clean["action"] != raw["action"] or clean["situation"] != raw["situation"]
        clean["redacted"] = bool(step.redacted or changed)
        stored = TraceStep.model_validate(clean)
        self._append(stored.model_dump_json())
        return stored

    def record_verdict(self, verdict: TraceVerdict) -> TraceVerdict:
        """Append a verdict record."""
        clean = TraceVerdict.model_validate(
            self._redactor.redact_obj(verdict.model_dump(mode="json"))
        )
        self._append(clean.model_dump_json())
        return clean

    def record_overhead(self, overhead: TraceOverhead) -> TraceOverhead:
        """Append an overhead record (nothing is written when nothing was spent)."""
        if overhead.usage == Usage():
            return overhead
        clean = TraceOverhead.model_validate(
            self._redactor.redact_obj(overhead.model_dump(mode="json"))
        )
        self._append(clean.model_dump_json())
        return clean


@dataclass
class RunRecords:
    """Parsed records of one run file."""

    steps: list[TraceStep] = field(default_factory=list)
    verdicts: list[TraceVerdict] = field(default_factory=list)
    overheads: list[TraceOverhead] = field(default_factory=list)


class TraceStore:
    """Reads the trace files of every run."""

    def __init__(self, directory: Path) -> None:
        """Read from ``directory``."""
        self.directory = directory

    def run_ids(self) -> list[str]:
        """Run ids with a trace file, in sorted (chronological) order."""
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.jsonl"))

    def records(self, run_id: str) -> RunRecords:
        """Parse one run's file, skipping unreadable (e.g. torn) lines."""
        path = self.directory / f"{run_id}.jsonl"
        parsed = RunRecords()
        if not path.is_file():
            return parsed
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                kind = data.get("type")
                if kind == "verdict":
                    parsed.verdicts.append(TraceVerdict.model_validate(data))
                elif kind == "overhead":
                    parsed.overheads.append(TraceOverhead.model_validate(data))
                else:
                    parsed.steps.append(TraceStep.model_validate(data))
            except (json.JSONDecodeError, ValidationError, AttributeError):
                _log.warning(
                    "skipping unreadable trace line", extra={"run": run_id, "line": number}
                )
        return parsed

    def load_run(self, run_id: str) -> list[TraceStep]:
        """Steps of one run with ``verified`` derived from the latest verdicts."""
        parsed = self.records(run_id)
        outcome: dict[tuple[str, int], bool] = {}
        for verdict in parsed.verdicts:
            outcome[(verdict.task_id, verdict.attempt)] = verdict.passed
        return [
            step.model_copy(update={"verified": outcome.get((step.task_id, step.attempt), False)})
            for step in parsed.steps
        ]

    def load_all(self) -> list[TraceStep]:
        """Steps of every run, in run order then file order."""
        return [step for run_id in self.run_ids() for step in self.load_run(run_id)]

    def run_usage(self, run_id: str) -> Usage:
        """Total usage recorded for one run (steps plus overheads)."""
        parsed = self.records(run_id)
        total = Usage()
        for step in parsed.steps:
            total = total.plus(step_usage(step))
        for overhead in parsed.overheads:
            total = total.plus(overhead.usage)
        return total

    def overheads(self) -> list[TraceOverhead]:
        """Every overhead record of every run."""
        return [item for run_id in self.run_ids() for item in self.records(run_id).overheads]


def step_usage(step: TraceStep) -> Usage:
    """Usage recorded on a step."""
    return Usage(
        tokens_in=step.tokens_in,
        tokens_out=step.tokens_out,
        cost=step.cost,
        model_calls=step.model_calls,
    )


def normalize(step: TraceStep) -> dict[str, Any]:
    """Canonical view of a step used for comparison: situation, normalized action, outcome."""
    action: Action = normalize_action(step.action)
    return {
        "situation": step.situation.model_dump(mode="json"),
        "action": action.model_dump(mode="json"),
        "ok": step.result.ok,
    }


def cost_report(
    steps: list[TraceStep],
    overheads: list[TraceOverhead],
    costs: CostTable,
    skills: dict[str, int],
    runs: int,
) -> CostReport:
    """Cost by route plus the estimated saving against an all-large baseline.

    The baseline prices every executed (non-replayed) step as one large-tier call: model-routed
    steps with their own per-call tokens, other steps with the mean per-call tokens of the
    model-routed steps (zero if there are none). This is an estimate and is labelled as such.
    """
    executed = [step for step in steps if not step.replayed]
    by_route: dict[str, RouteCost] = defaultdict(RouteCost)
    for step in executed:
        current = by_route[step.route]
        by_route[step.route] = current.model_copy(
            update={
                "steps": current.steps + 1,
                "tokens_in": current.tokens_in + step.tokens_in,
                "tokens_out": current.tokens_out + step.tokens_out,
                "cost": current.cost + step.cost,
                "model_calls": current.model_calls + step.model_calls,
            }
        )
    overhead = Usage()
    for item in overheads:
        overhead = overhead.plus(item.usage)
    total = overhead
    for step in executed:
        total = total.plus(step_usage(step))
    per_call = [
        (step.tokens_in / step.model_calls, step.tokens_out / step.model_calls)
        for step in executed
        if step.model_calls > 0
    ]
    mean_in = sum(p[0] for p in per_call) / len(per_call) if per_call else 0.0
    mean_out = sum(p[1] for p in per_call) / len(per_call) if per_call else 0.0
    baseline = 0.0
    for step in executed:
        if step.model_calls > 0:
            tokens_in = step.tokens_in / step.model_calls
            tokens_out = step.tokens_out / step.model_calls
        else:
            tokens_in, tokens_out = mean_in, mean_out
        baseline += costs.cost("large", round(tokens_in), round(tokens_out))
    saving = baseline - total.cost
    return CostReport(
        runs=runs,
        steps=len(executed),
        by_route=dict(sorted(by_route.items())),
        overhead=overhead,
        total=total,
        skills=skills,
        baseline_cost=round(baseline, 9),
        estimated_saving=round(saving, 9),
        saving_pct=round(100 * saving / baseline, 3) if baseline else 0.0,
        note="baseline is an estimate: every executed step priced as one large-tier call",
    )
