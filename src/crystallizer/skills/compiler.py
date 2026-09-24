"""Compile a mined pattern into a declarative skill (JSON) with a guard. Never generates code.

Slots: every placeholder field becomes a slot bound to ``params.<field>``; its type is inferred
from the observed values (int, then identifier, then path, then string). The guard is true only
inside the observed domain:

* ``eq`` on ``task_kind`` and on the first step's ``step_kind``;
* ``range`` on ``step_index`` and on numeric non-slot params (observed min and max);
* ``eq``/``in`` on categorical non-slot params present in every occurrence;
* ``has_type`` on slot params;
* ``exists`` on params present in every occurrence whose values mix types.

The skill id is ``skill-`` plus 12 hex digits of the SHA-256 of ``(task_kind, steps, slots)``, so
re-mining the same behavior yields the same id; the content hash covers the whole skill.
"""

from __future__ import annotations

import json
from pathlib import Path

from crystallizer.checkpoint import atomic_write
from crystallizer.hashing import digest
from crystallizer.schemas import (
    Condition,
    Guard,
    GuardOp,
    RangeValue,
    Scalar,
    Skill,
    Slot,
    SlotType,
)
from crystallizer.skills.executor import check_type, validate_skill
from crystallizer.skills.miner import Pattern

SLOT_TYPE_ORDER = (SlotType.INT, SlotType.IDENTIFIER, SlotType.PATH, SlotType.STRING)


def infer_slot_type(values: list[Scalar]) -> SlotType | None:
    """Most specific slot type satisfied by every value, or None."""
    for slot_type in SLOT_TYPE_ORDER:
        if values and all(check_type(value, slot_type) for value in values):
            return slot_type
    return None


def _is_number(value: Scalar) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _sort_key(value: Scalar) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def build_guard(pattern: Pattern, slots: dict[str, Slot]) -> Guard:
    """Guard that is true only inside the pattern's observed domain."""
    situations = [occ.situation for occ in pattern.occurrences]
    indices = [situation.step_index for situation in situations]
    conditions = [
        Condition(field="task_kind", op=GuardOp.EQ, value=pattern.task_kind),
        Condition(field="step_kind", op=GuardOp.EQ, value=pattern.steps[0].step_kind),
        Condition(
            field="step_index",
            op=GuardOp.RANGE,
            value=RangeValue(min=min(indices), max=max(indices)),
        ),
    ]
    names = sorted({name for situation in situations for name in situation.params})
    for name in names:
        field = f"params.{name}"
        values = [s.params[name] for s in situations if name in s.params]
        if name in slots:
            conditions.append(
                Condition(field=field, op=GuardOp.HAS_TYPE, value=slots[name].type.value)
            )
            continue
        if len(values) != len(situations):
            continue
        if all(_is_number(value) for value in values):
            conditions.append(
                Condition(
                    field=field,
                    op=GuardOp.RANGE,
                    value=RangeValue(min=float(min(values)), max=float(max(values))),
                )
            )
        elif all(isinstance(value, str) for value in values) or all(
            isinstance(value, bool) for value in values
        ):
            unique = sorted(set(values), key=_sort_key)
            if len(unique) == 1:
                conditions.append(Condition(field=field, op=GuardOp.EQ, value=unique[0]))
            else:
                conditions.append(Condition(field=field, op=GuardOp.IN, value=unique))
        else:
            conditions.append(Condition(field=field, op=GuardOp.EXISTS))
    return Guard(all=conditions)


def guard_summary(guard: Guard) -> str:
    """Human-readable guard."""
    parts: list[str] = []
    for condition in guard.all:
        value = condition.value
        if condition.op is GuardOp.EQ:
            parts.append(f"{condition.field} == {json.dumps(value)}")
        elif condition.op is GuardOp.IN and isinstance(value, list):
            parts.append(f"{condition.field} in {json.dumps(value)}")
        elif condition.op is GuardOp.RANGE and isinstance(value, RangeValue):
            parts.append(f"{value.min:g} <= {condition.field} <= {value.max:g}")
        elif condition.op is GuardOp.HAS_TYPE:
            parts.append(f"{condition.field} is {value}")
        else:
            parts.append(f"{condition.field} exists")
    return " AND ".join(parts)


def skill_id(task_kind: str, skill_dump: dict[str, object]) -> str:
    """Stable id from what the skill does (not from its guard)."""
    core = {"task_kind": task_kind, "steps": skill_dump["steps"], "slots": skill_dump["slots"]}
    return "skill-" + digest(core)[:12]


def content_hash(skill: Skill) -> str:
    """Hash of the full skill content except its version."""
    data = skill.model_dump(mode="json")
    data.pop("version", None)
    return digest(data)


def compile_pattern(pattern: Pattern, version: int = 1) -> Skill | None:
    """Compile ``pattern``; None when a slot has no consistent type."""
    slots: dict[str, Slot] = {}
    for name in pattern.slot_fields:
        values = [
            occ.situation.params[name]
            for occ in pattern.occurrences
            if name in occ.situation.params
        ]
        if len(values) != len(pattern.occurrences):
            return None
        slot_type = infer_slot_type(values)
        if slot_type is None:
            return None
        slots[name] = Slot(type=slot_type, source=f"params.{name}")
    guard = build_guard(pattern, slots)
    draft = Skill(
        id="skill-000000000000",
        version=version,
        task_kind=pattern.task_kind,
        slots=slots,
        guard=guard,
        steps=pattern.steps,
    )
    identifier = skill_id(pattern.task_kind, draft.model_dump(mode="json"))
    return validate_skill(draft.model_copy(update={"id": identifier}))


def skill_filename(skill: Skill) -> str:
    """File name of a skill version."""
    return f"{skill.id}.v{skill.version}.json"


def write_skill_file(skill: Skill, directory: Path) -> Path:
    """Write a skill JSON file atomically and return its path."""
    path = directory / skill_filename(skill)
    atomic_write(path, json.dumps(skill.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return path
