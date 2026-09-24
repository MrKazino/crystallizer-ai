"""The only interpreter of skills. Skills are data: nothing here evaluates, executes or imports.

Responsibilities:

* parse and validate skill JSON (unknown ops, filters, slots, malformed or unresolved
  placeholders, and slot misuse are all rejected);
* evaluate a guard against a situation (a missing field makes a condition false);
* bind slots from their situation sources and check their types;
* render templated arguments into concrete :class:`~crystallizer.schemas.Action` objects.

Template syntax inside string arguments: ``{slot}`` or ``{slot|filter}`` with filters ``lower``,
``upper`` and ``snake``; ``{{`` and ``}}`` are literal braces.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from crystallizer.errors import SkillError
from crystallizer.schemas import (
    Action,
    ArgValue,
    Condition,
    Guard,
    GuardOp,
    RangeValue,
    Scalar,
    Situation,
    Skill,
    SlotType,
)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_PATH = re.compile(r"[A-Za-z0-9_.\-/]{1,255}")
_SLOT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
MAX_STRING = 1024


def to_snake(text: str) -> str:
    """``ParserModule`` -> ``parser_module``; ``parser-module`` -> ``parser_module``."""
    step = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    step = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step)
    return step.replace("-", "_").lower()


FILTERS: dict[str, Callable[[str], str]] = {
    "lower": str.lower,
    "upper": str.upper,
    "snake": to_snake,
}


def check_type(value: Scalar | None, slot_type: SlotType) -> bool:
    """True if ``value`` is a valid instance of ``slot_type``."""
    if slot_type is SlotType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if not isinstance(value, str):
        return False
    if slot_type is SlotType.IDENTIFIER:
        return _IDENTIFIER.fullmatch(value) is not None
    if slot_type is SlotType.PATH:
        if _PATH.fullmatch(value) is None or value.startswith("/"):
            return False
        return all(part not in ("", "..") for part in value.split("/"))
    return len(value) <= MAX_STRING and value.isprintable()


@dataclass(frozen=True)
class Placeholder:
    """A parsed ``{slot|filter}``."""

    slot: str
    filter: str | None


Token = str | Placeholder


def parse_template(text: str) -> list[Token]:
    """Split a template into literal strings and placeholders; reject malformed syntax."""
    tokens: list[Token] = []
    literal: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "{":
            if text.startswith("{{", index):
                literal.append("{")
                index += 2
                continue
            end = text.find("}", index + 1)
            if end == -1:
                raise SkillError(f"unterminated placeholder in {text!r}")
            inner = text[index + 1 : end]
            name, _, filter_name = inner.partition("|")
            if not _SLOT_NAME.fullmatch(name):
                raise SkillError(f"malformed placeholder {{{inner}}}")
            if "|" in inner and filter_name not in FILTERS:
                raise SkillError(f"unknown filter {filter_name!r}")
            if literal:
                tokens.append("".join(literal))
                literal = []
            tokens.append(Placeholder(name, filter_name or None))
            index = end + 1
        elif char == "}":
            if not text.startswith("}}", index):
                raise SkillError(f"unmatched '}}' in {text!r}")
            literal.append("}")
            index += 2
        else:
            literal.append(char)
            index += 1
    if literal:
        tokens.append("".join(literal))
    return tokens


def escape_literal(text: str) -> str:
    """Escape braces so ``text`` renders to itself."""
    return text.replace("{", "{{").replace("}", "}}")


def render_template(text: str, values: Mapping[str, Scalar]) -> str:
    """Render a template with slot values."""
    pieces: list[str] = []
    for token in parse_template(text):
        if isinstance(token, str):
            pieces.append(token)
            continue
        if token.slot not in values:
            raise SkillError(f"unresolved placeholder {{{token.slot}}}")
        value = str(values[token.slot])
        pieces.append(FILTERS[token.filter](value) if token.filter else value)
    return "".join(pieces)


def _templates(value: ArgValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return list(value)
    return []


def validate_skill(skill: Skill) -> Skill:
    """Check every template and slot reference; return the skill."""
    for step in skill.steps:
        for value in step.args.values():
            for template in _templates(value):
                for token in parse_template(template):
                    if isinstance(token, str):
                        continue
                    slot = skill.slots.get(token.slot)
                    if slot is None:
                        raise SkillError(f"unknown slot {token.slot!r} in step {step.step_kind}")
                    if token.filter and slot.type is SlotType.INT:
                        raise SkillError(f"filter {token.filter!r} cannot apply to int slot")
    return skill


def parse_skill(data: str | bytes | Mapping[str, Any]) -> Skill:
    """Parse and fully validate skill JSON (or an already-decoded mapping)."""
    try:
        if isinstance(data, str | bytes):
            skill = Skill.model_validate_json(data)
        else:
            skill = Skill.model_validate(dict(data))
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise SkillError(f"invalid skill: {details}") from None
    return validate_skill(skill)


def _same(left: Scalar | None, right: Scalar) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    return isinstance(left, str) and isinstance(right, str) and left == right


def condition_holds(condition: Condition, situation: Situation) -> bool:
    """Evaluate one condition. A missing field is always false."""
    exists, value = situation.lookup(condition.field)
    if not exists:
        return False
    op = condition.op
    expected = condition.value
    if op is GuardOp.EXISTS:
        return True
    if op is GuardOp.EQ:
        return isinstance(expected, str | int | float | bool) and _same(value, expected)
    if op is GuardOp.IN:
        return isinstance(expected, list) and any(_same(value, item) for item in expected)
    if op is GuardOp.RANGE:
        if not isinstance(expected, RangeValue) or isinstance(value, bool):
            return False
        return isinstance(value, int | float) and expected.min <= value <= expected.max
    return isinstance(expected, str) and check_type(value, SlotType(expected))


def evaluate_guard(guard: Guard, situation: Situation) -> bool:
    """True only if every condition holds."""
    return all(condition_holds(condition, situation) for condition in guard.all)


def bind_slots(skill: Skill, situation: Situation) -> dict[str, Scalar]:
    """Read every slot from its source field and check its type."""
    values: dict[str, Scalar] = {}
    for name, slot in skill.slots.items():
        exists, value = situation.lookup(slot.source)
        if not exists or value is None:
            raise SkillError(f"slot {name!r}: situation has no {slot.source}")
        if not check_type(value, slot.type):
            raise SkillError(f"slot {name!r}: value does not have type {slot.type.value}")
        values[name] = value
    return values


def _render_arg(value: ArgValue, values: Mapping[str, Scalar]) -> ArgValue:
    if isinstance(value, str):
        return render_template(value, values)
    if isinstance(value, list):
        return [render_template(item, values) for item in value]
    return value


def instantiate(skill: Skill, situation: Situation) -> list[Action]:
    """Return the concrete actions of ``skill`` for ``situation`` (guard must hold)."""
    if not evaluate_guard(skill.guard, situation):
        raise SkillError(f"guard of {skill.key} is false for this situation")
    values = bind_slots(skill, situation)
    return [
        Action(
            tool=step.tool,
            args={name: _render_arg(arg, values) for name, arg in step.args.items()},
        )
        for step in skill.steps
    ]
