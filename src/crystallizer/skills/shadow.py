"""Shadow-test skills against verified outcomes without executing anything.

A skill is evaluated only on occurrences that were NOT used to derive it: steps outside its
provenance, from runs outside its ``derived_runs`` (i.e. later runs). For every verified task
attempt, at each position where the skill's step kinds match, the guard holds and no step is
redacted, the executor's actions are compared with the recorded, normalized actions. A match is
a pass. An *unsafe diff* is an emitted action classified irreversible where the verified action
was reversible or absent; any unsafe diff blocks promotion.

Observations and live runs are stored with unique keys, so evaluating the same trace twice
(online during a run and offline via ``skills evaluate``) never double counts.
"""

from __future__ import annotations

from collections.abc import Sequence

from crystallizer.clock import Clock, iso
from crystallizer.db import Database
from crystallizer.errors import SkillError
from crystallizer.policy import Policy
from crystallizer.schemas import Action, Skill, Strict, TraceStep
from crystallizer.skills.executor import evaluate_guard, instantiate
from crystallizer.tools import normalize_action


class ShadowObservation(Strict):
    """One shadow comparison."""

    skill_key: str
    occurrence: str
    passed: bool
    unsafe: bool


def occurrence_key(step: TraceStep) -> str:
    """Unique position of a step: ``run:task:attempt:index``."""
    return f"{step.run_id}:{step.task_id}:{step.attempt}:{step.index}"


def _unsafe(emitted: Sequence[Action], actual: Sequence[Action], policy: Policy) -> bool:
    for position, action in enumerate(emitted):
        if not policy.classify(action).irreversible:
            continue
        if position >= len(actual) or not policy.classify(actual[position]).irreversible:
            return True
    return False


def shadow_evaluate(
    skill: Skill,
    steps: Sequence[TraceStep],
    *,
    provenance: set[str],
    derived_runs: set[str],
    policy: Policy,
) -> list[ShadowObservation]:
    """Compare ``skill`` with one verified attempt's steps (ordered by index)."""
    kinds = [step.step_kind for step in skill.steps]
    size = len(kinds)
    observations: list[ShadowObservation] = []
    for start, first in enumerate(steps):
        window = list(steps[start : start + size])
        if len(window) < size or first.run_id in derived_runs:
            continue
        if any(step.index != first.index + offset for offset, step in enumerate(window)):
            continue
        if [step.situation.step_kind for step in window] != kinds:
            continue
        if any(step.id in provenance or step.redacted for step in window):
            continue
        if not evaluate_guard(skill.guard, first.situation):
            continue
        actual = [normalize_action(step.action) for step in window]
        try:
            emitted = [normalize_action(action) for action in instantiate(skill, first.situation)]
        except SkillError:
            emitted = []
        observations.append(
            ShadowObservation(
                skill_key=skill.key,
                occurrence=occurrence_key(first),
                passed=emitted == actual,
                unsafe=_unsafe(emitted, actual, policy),
            )
        )
    return observations


class EvidenceStore:
    """Shadow observations and live runs in the state database."""

    def __init__(self, db: Database, clock: Clock) -> None:
        """Bind to ``db``."""
        self._db = db
        self._clock = clock

    def record_shadow(self, observations: Sequence[ShadowObservation]) -> int:
        """Insert observations (existing keys are ignored); return how many were new."""
        added = 0
        for obs in observations:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO shadow_obs (skill_key, occurrence, passed, unsafe, ts)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    obs.skill_key,
                    obs.occurrence,
                    int(obs.passed),
                    int(obs.unsafe),
                    iso(self._clock.now()),
                ),
            )
            added += cursor.rowcount
        return added

    def shadow_stats(self, skill_key: str) -> tuple[int, int, int]:
        """``(runs, passes, unsafe_diffs)`` for a skill version."""
        row = self._db.query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(passed), 0) AS p, COALESCE(SUM(unsafe), 0) AS u"
            " FROM shadow_obs WHERE skill_key = ?",
            (skill_key,),
        )[0]
        return int(row["n"]), int(row["p"]), int(row["u"])

    def record_live(self, skill_key: str, occurrence: str, passed: bool) -> bool:
        """Record a live run (idempotent); return True if it was new."""
        cursor = self._db.execute(
            "INSERT OR IGNORE INTO live_runs (skill_key, occurrence, passed, ts)"
            " VALUES (?, ?, ?, ?)",
            (skill_key, occurrence, int(passed), iso(self._clock.now())),
        )
        return cursor.rowcount > 0

    def live_stats(self, skill_key: str) -> tuple[int, int]:
        """``(runs, passes)`` of live use."""
        row = self._db.query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(passed), 0) AS p"
            " FROM live_runs WHERE skill_key = ?",
            (skill_key,),
        )[0]
        return int(row["n"]), int(row["p"])

    def live_window(self, skill_key: str, size: int) -> list[bool]:
        """The most recent ``size`` live outcomes, newest first."""
        rows = self._db.query(
            "SELECT passed FROM live_runs WHERE skill_key = ? ORDER BY id DESC LIMIT ?",
            (skill_key, size),
        )
        return [bool(row["passed"]) for row in rows]
