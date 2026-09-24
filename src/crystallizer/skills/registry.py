"""Versioned skill store with evidence-based promotion and automatic demotion.

Layout under ``state_dir/skills/``: ``candidates/``, ``active/``, ``demoted/``, ``manifest.json``.
Only :meth:`SkillRegistry.promote` moves a skill into ``active/``. Promotion requires
``shadow_runs >= min_shadow_runs``, ``pass_rate >= promote_pass_rate`` and zero unsafe diffs
(``--force`` needs a reason and is recorded). A skill is demoted when its rolling live pass rate
over the last ``demote_window`` runs (or all of them, while fewer exist) drops below
``demote_pass_rate``, or immediately when it emits an irreversible action. Every promotion and
demotion is recorded in the manifest history with its evidence, including the Wilson 95% lower
bound of the pass rate (informational).

Anti-flapping: a pattern whose content equals a demoted version is never re-added; a changed
pattern for a demoted id becomes a new version; candidate and active ids are never churned.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crystallizer.checkpoint import atomic_write
from crystallizer.clock import Clock, iso
from crystallizer.config import SkillsConfig
from crystallizer.db import Database
from crystallizer.errors import SkillError, UsageError
from crystallizer.events import EventBus, EventKind
from crystallizer.policy import Policy
from crystallizer.schemas import (
    HistoryEvent,
    RegistryManifest,
    Scalar,
    Skill,
    SkillManifest,
    SkillStatus,
    TraceStep,
)
from crystallizer.skills.compiler import (
    compile_pattern,
    content_hash,
    guard_summary,
    skill_filename,
    write_skill_file,
)
from crystallizer.skills.executor import parse_skill
from crystallizer.skills.miner import attempt_groups, mine
from crystallizer.skills.shadow import EvidenceStore, shadow_evaluate
from crystallizer.traces import TraceStore

STATUS_DIRS = {
    SkillStatus.CANDIDATE: "candidates",
    SkillStatus.ACTIVE: "active",
    SkillStatus.DEMOTED: "demoted",
}


def wilson_lower(passes: int, runs: int, z: float = 1.96) -> float:
    """Wilson score lower bound of a pass rate (0 when there are no runs)."""
    if runs == 0:
        return 0.0
    phat = passes / runs
    denominator = 1 + z * z / runs
    centre = phat + z * z / (2 * runs)
    margin = z * math.sqrt(phat * (1 - phat) / runs + z * z / (4 * runs * runs))
    return round((centre - margin) / denominator, 6)


def load_manifest(skills_dir: Path) -> RegistryManifest:
    """Read ``manifest.json`` (empty manifest when absent). Read-only; safe for dry runs."""
    path = skills_dir / "manifest.json"
    if not path.is_file():
        return RegistryManifest()
    return RegistryManifest.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_reference(manifest: RegistryManifest, reference: str) -> SkillManifest:
    """Resolve ``skill-id`` (latest version) or ``skill-id@vN`` in ``manifest``."""
    matches = [s for s in manifest.skills if reference in (s.key, s.id)]
    if not matches:
        raise UsageError(f"unknown skill {reference!r}")
    return max(matches, key=lambda s: s.version)


def read_skill(skills_dir: Path, entry: SkillManifest) -> Skill:
    """Load and validate the skill file of ``entry`` (read-only)."""
    return parse_skill((skills_dir / entry.source_file).read_text(encoding="utf-8"))


class SkillRegistry:
    """The skill registry for one state directory."""

    def __init__(
        self,
        state_dir: Path,
        db: Database,
        config: SkillsConfig,
        clock: Clock,
        policy: Policy,
        bus: EventBus | None = None,
    ) -> None:
        """Open (creating directories as needed) the registry."""
        self.directory = state_dir / "skills"
        for name in STATUS_DIRS.values():
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self._config = config
        self._clock = clock
        self._policy = policy
        self._bus = bus
        self.evidence = EvidenceStore(db, clock)
        self.manifest = load_manifest(self.directory)

    # ------------------------------------------------------------------ persistence

    @property
    def version(self) -> int:
        """Registry version (incremented on every change)."""
        return self.manifest.registry_version

    def _save(self) -> None:
        self.manifest = self.manifest.model_copy(
            update={"registry_version": self.manifest.registry_version + 1}
        )
        text = json.dumps(self.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        atomic_write(self.directory / "manifest.json", text)

    def _replace(self, entry: SkillManifest) -> None:
        skills = [entry if s.key == entry.key else s for s in self.manifest.skills]
        if entry.key not in {s.key for s in self.manifest.skills}:
            skills.append(entry)
        skills.sort(key=lambda s: (s.id, s.version))
        self.manifest = self.manifest.model_copy(update={"skills": skills})

    def _event(self, kind: EventKind, entry: SkillManifest, reason: str) -> None:
        if self._bus is not None:
            self._bus.publish(kind, skill=entry.key, status=entry.status.value, reason=reason)

    def _history(
        self, event: str, reason: str, evidence: dict[str, Scalar], forced: bool = False
    ) -> HistoryEvent:
        return HistoryEvent.model_validate(
            {
                "ts": iso(self._clock.now()),
                "event": event,
                "reason": reason,
                "forced": forced,
                "evidence": evidence,
            }
        )

    # ------------------------------------------------------------------ lookup

    def entries(self, status: SkillStatus | None = None) -> list[SkillManifest]:
        """Manifest entries, optionally filtered by status."""
        return [s for s in self.manifest.skills if status is None or s.status is status]

    def get(self, reference: str) -> SkillManifest:
        """Resolve ``skill-id`` (latest version) or ``skill-id@vN``."""
        return resolve_reference(self.manifest, reference)

    def load(self, entry: SkillManifest) -> Skill:
        """Load and validate a skill file."""
        return read_skill(self.directory, entry)

    def counts(self) -> dict[str, int]:
        """Number of skills per status."""
        counts = {status.value: 0 for status in SkillStatus}
        for entry in self.manifest.skills:
            counts[entry.status.value] += 1
        return counts

    def active_skills(self) -> list[tuple[Skill, SkillManifest]]:
        """Active skills in precedence order: most guard conditions, most steps, pass rate, id."""
        pairs = [(self.load(entry), entry) for entry in self.entries(SkillStatus.ACTIVE)]
        pairs.sort(
            key=lambda p: (-len(p[0].guard.all), -len(p[0].steps), -p[1].pass_rate, p[1].key)
        )
        return pairs

    # ------------------------------------------------------------------ candidates

    def add_candidate(
        self,
        skill: Skill,
        *,
        provenance: Sequence[str] = (),
        derived_runs: Sequence[str] = (),
        origin: str = "mined",
    ) -> SkillManifest | None:
        """Register a new candidate version, or return None if it would churn or flap."""
        digest_ = content_hash(skill)
        versions = [s for s in self.manifest.skills if s.id == skill.id]
        if any(s.content_hash == digest_ for s in versions):
            return None
        if any(s.status in (SkillStatus.CANDIDATE, SkillStatus.ACTIVE) for s in versions):
            return None
        version = max((s.version for s in versions), default=0) + 1
        skill = skill.model_copy(update={"version": version})
        write_skill_file(skill, self.directory / STATUS_DIRS[SkillStatus.CANDIDATE])
        event = "imported" if origin == "imported" else "created"
        entry = SkillManifest.model_validate(
            {
                "id": skill.id,
                "version": version,
                "content_hash": digest_,
                "guard_summary": guard_summary(skill.guard),
                "source_file": f"{STATUS_DIRS[SkillStatus.CANDIDATE]}/{skill_filename(skill)}",
                "provenance": list(provenance),
                "derived_runs": list(derived_runs),
                "origin": origin,
                "created_at": iso(self._clock.now()),
                "history": [
                    self._history(event, f"{origin} candidate", {"steps": len(skill.steps)})
                ],
            }
        )
        self._replace(entry)
        self._save()
        self._event(EventKind.SKILL_MINED, entry, origin)
        return entry

    def mine(self, store: TraceStore) -> list[SkillManifest]:
        """Mine all recorded traces and register new candidates."""
        patterns = mine(
            store.load_all(),
            min_repeats=self._config.min_repeats,
            min_template_length=self._config.min_template_length,
        )
        added: list[SkillManifest] = []
        for pattern in patterns:
            skill = compile_pattern(pattern)
            if skill is None:
                continue
            entry = self.add_candidate(
                skill, provenance=pattern.provenance, derived_runs=pattern.runs
            )
            if entry is not None:
                added.append(entry)
        return added

    # ------------------------------------------------------------------ evidence

    def _refreshed(self, entry: SkillManifest) -> SkillManifest:
        runs, passes, unsafe = self.evidence.shadow_stats(entry.key)
        live_runs, live_passes = self.evidence.live_stats(entry.key)
        return entry.model_copy(
            update={
                "shadow_runs": runs,
                "shadow_passes": passes,
                "unsafe_diffs": unsafe,
                "pass_rate": round(passes / runs, 6) if runs else 0.0,
                "live_runs": live_runs,
                "live_passes": live_passes,
            }
        )

    def observe(self, steps: Sequence[TraceStep]) -> int:
        """Shadow-evaluate every candidate on one verified attempt; return new observations."""
        ordered = sorted(steps, key=lambda s: s.index)
        added = 0
        changed = False
        for entry in self.entries(SkillStatus.CANDIDATE):
            skill = self.load(entry)
            observations = shadow_evaluate(
                skill,
                ordered,
                provenance=set(entry.provenance),
                derived_runs=set(entry.derived_runs),
                policy=self._policy,
            )
            new = self.evidence.record_shadow(observations)
            if new:
                added += new
                self._replace(self._refreshed(entry))
                changed = True
        if changed:
            self._save()
        return added

    def evaluate(self, store: TraceStore) -> int:
        """Offline shadow evaluation over every verified attempt in the traces."""
        added = 0
        for group in attempt_groups(store.load_all()).values():
            added += self.observe(group)
        return added

    def record_live(self, skill_key: str, occurrence: str, passed: bool) -> SkillManifest | None:
        """Record live use of an active skill; demote it if its rolling pass rate drops."""
        entry = self.get(skill_key)
        if not self.evidence.record_live(entry.key, occurrence, passed):
            return None
        entry = self._refreshed(entry)
        self._replace(entry)
        window = self.evidence.live_window(entry.key, self._config.demote_window)
        rate = sum(window) / len(window)
        if entry.status is SkillStatus.ACTIVE and rate < self._config.demote_pass_rate:
            evidence: dict[str, Scalar] = {
                "window": len(window),
                "window_pass_rate": round(rate, 6),
                "threshold": self._config.demote_pass_rate,
                "live_runs": entry.live_runs,
            }
            return self._move(
                entry, SkillStatus.DEMOTED, "demoted", "rolling live pass rate", evidence
            )
        self._save()
        return None

    def demote_unsafe(self, skill_key: str, reason: str) -> SkillManifest:
        """Immediately demote an active skill that emitted an unsafe action."""
        entry = self.get(skill_key)
        return self._move(entry, SkillStatus.DEMOTED, "demoted", reason, {"unsafe": True})

    # ------------------------------------------------------------------ lifecycle

    def promotion_check(self, entry: SkillManifest) -> tuple[bool, str, dict[str, Scalar]]:
        """Whether ``entry`` meets the promotion criteria, why, and the evidence."""
        entry = self._refreshed(entry)
        evidence: dict[str, Scalar] = {
            "shadow_runs": entry.shadow_runs,
            "shadow_passes": entry.shadow_passes,
            "pass_rate": entry.pass_rate,
            "unsafe_diffs": entry.unsafe_diffs,
            "wilson_lower_95": wilson_lower(entry.shadow_passes, entry.shadow_runs),
            "min_shadow_runs": self._config.min_shadow_runs,
            "promote_pass_rate": self._config.promote_pass_rate,
        }
        if entry.unsafe_diffs:
            return False, f"{entry.unsafe_diffs} unsafe diff(s)", evidence
        if entry.shadow_runs < self._config.min_shadow_runs:
            return (
                False,
                (f"only {entry.shadow_runs} shadow run(s), need {self._config.min_shadow_runs}"),
                evidence,
            )
        if entry.pass_rate < self._config.promote_pass_rate:
            return (
                False,
                (f"pass rate {entry.pass_rate:.3f} below {self._config.promote_pass_rate}"),
                evidence,
            )
        return True, "shadow evidence meets the promotion criteria", evidence

    def promote(
        self, reference: str, *, force: bool = False, reason: str | None = None
    ) -> SkillManifest:
        """Move a candidate to ``active/`` if the evidence allows (or if forced with a reason)."""
        entry = self.get(reference)
        if entry.status is not SkillStatus.CANDIDATE:
            raise SkillError(
                f"{entry.key} is {entry.status.value}; only candidates can be promoted"
            )
        if force and not (reason and reason.strip()):
            raise UsageError("--force requires --reason")
        ok, why, evidence = self.promotion_check(entry)
        if not ok and not force:
            raise SkillError(f"promotion of {entry.key} refused: {why}")
        text = f"forced: {reason}" if force and not ok else (reason or why)
        return self._move(
            entry, SkillStatus.ACTIVE, "promoted", text, evidence, forced=force and not ok
        )

    def demote(self, reference: str, *, reason: str) -> SkillManifest:
        """Manually demote an active skill."""
        entry = self.get(reference)
        if entry.status is not SkillStatus.ACTIVE:
            raise SkillError(
                f"{entry.key} is {entry.status.value}; only active skills can be demoted"
            )
        live_runs, live_passes = self.evidence.live_stats(entry.key)
        evidence: dict[str, Scalar] = {"live_runs": live_runs, "live_passes": live_passes}
        return self._move(entry, SkillStatus.DEMOTED, "demoted", reason, evidence)

    def try_promote_all(self) -> list[SkillManifest]:
        """Promote every candidate that meets the criteria."""
        promoted: list[SkillManifest] = []
        for entry in sorted(self.entries(SkillStatus.CANDIDATE), key=lambda s: s.key):
            ok, _, _ = self.promotion_check(entry)
            if ok:
                promoted.append(self.promote(entry.key))
        return promoted

    def _move(
        self,
        entry: SkillManifest,
        status: SkillStatus,
        event: str,
        reason: str,
        evidence: dict[str, Scalar],
        forced: bool = False,
    ) -> SkillManifest:
        skill = self.load(entry)
        old = self.directory / entry.source_file
        write_skill_file(skill, self.directory / STATUS_DIRS[status])
        entry = self._refreshed(entry).model_copy(
            update={
                "status": status,
                "source_file": f"{STATUS_DIRS[status]}/{skill_filename(skill)}",
                "history": [*entry.history, self._history(event, reason, evidence, forced)],
            }
        )
        self._replace(entry)
        self._save()
        if old != self.directory / entry.source_file:
            old.unlink(missing_ok=True)
        kind = EventKind.SKILL_PROMOTED if status is SkillStatus.ACTIVE else EventKind.SKILL_DEMOTED
        self._event(kind, entry, reason)
        return entry

    # ------------------------------------------------------------------ interchange

    def export_skill(self, reference: str) -> dict[str, Any]:
        """The skill JSON of ``reference``."""
        return self.load(self.get(reference)).model_dump(mode="json")

    def import_skill(self, data: str | bytes | dict[str, Any]) -> SkillManifest | None:
        """Validate foreign skill JSON and add it as a candidate (it must earn promotion here)."""
        skill = parse_skill(data)
        return self.add_candidate(skill, origin="imported")
