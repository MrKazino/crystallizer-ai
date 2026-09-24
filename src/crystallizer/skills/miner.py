"""Find repeated mechanical sub-trajectories in verified traces.

1. **Eligible steps**: verified, successful, not redacted, and not routed to a skill (skills never
   reinforce themselves). Replayed duplicates keep one record per step position.
2. **Templating**: each string argument is rewritten against the situation's params: values (and
   their ``lower``/``upper``/``snake`` forms) of at least ``min_template_length`` characters,
   matched on non-alphanumeric boundaries, longest first, become ``{field}`` placeholders. The
   template must render back to the exact original, otherwise the literal is kept.
3. **Mechanical**: a step's signature is ``(step_kind, tool, templated args)``. It is mechanical
   iff the signature occurs in at least ``min_repeats`` distinct tasks: constant and
   situation-derived arguments give identical signatures; model-authored content does not.
4. **Patterns**: maximal runs of consecutive mechanical steps, keyed by task kind and signature
   sequence, kept when they occur in at least ``min_repeats`` distinct tasks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import Field

from crystallizer.hashing import canonical_json, digest
from crystallizer.schemas import ArgValue, Scalar, Situation, SkillStep, Strict, TraceStep
from crystallizer.skills.executor import (
    FILTERS,
    Placeholder,
    escape_literal,
    parse_template,
    render_template,
)

FILTER_ORDER = ("", "lower", "upper", "snake")


class Occurrence(Strict):
    """Where a pattern occurred: one contiguous run of steps in one task attempt."""

    run_id: str
    task_id: str
    attempt: int
    start: int
    step_ids: list[str]
    situation: Situation


class Pattern(Strict):
    """A repeated mechanical step sequence."""

    key: str
    task_kind: str
    steps: list[SkillStep]
    slot_fields: list[str] = Field(default_factory=list)
    occurrences: list[Occurrence]
    support: int

    @property
    def provenance(self) -> list[str]:
        """Every trace step id the pattern was derived from."""
        return [step_id for occ in self.occurrences for step_id in occ.step_ids]

    @property
    def runs(self) -> list[str]:
        """Runs the pattern was derived from."""
        return sorted({occ.run_id for occ in self.occurrences})


def _boundary(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not text[start - 1].isalnum()
    after_ok = end == len(text) or not text[end].isalnum()
    return before_ok and after_ok


def _candidates(situation: Situation, min_length: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    found: list[tuple[str, str, int]] = []
    order = 0
    for name, raw in sorted(situation.params.items()):
        if isinstance(raw, bool) or not isinstance(raw, str | int):
            continue
        base = str(raw)
        for filter_name in FILTER_ORDER if isinstance(raw, str) else ("",):
            rendered = FILTERS[filter_name](base) if filter_name else base
            if len(rendered) < min_length or rendered in seen:
                continue
            seen.add(rendered)
            suffix = f"|{filter_name}" if filter_name else ""
            found.append((rendered, f"{{{name}{suffix}}}", order))
            order += 1
    found.sort(key=lambda item: (-len(item[0]), item[2]))
    return [(rendered, placeholder) for rendered, placeholder, _ in found]


def templatize(value: str, situation: Situation, min_length: int = 3) -> str:
    """Rewrite ``value`` with placeholders for situation-derived substrings (exact round trip)."""
    candidates = _candidates(situation, min_length)
    pieces: list[str] = []
    index = 0
    while index < len(value):
        for rendered, placeholder in candidates:
            end = index + len(rendered)
            if value.startswith(rendered, index) and _boundary(value, index, end):
                pieces.append(placeholder)
                index = end
                break
        else:
            pieces.append(escape_literal(value[index]))
            index += 1
    template = "".join(pieces)
    values: dict[str, Scalar] = dict(situation.params)
    if render_template(template, values) != value:
        return escape_literal(value)
    return template


def templatize_args(
    args: dict[str, ArgValue], situation: Situation, min_length: int
) -> dict[str, ArgValue]:
    """Templatize every string (and list-of-string) argument."""
    result: dict[str, ArgValue] = {}
    for name, value in args.items():
        if isinstance(value, str):
            result[name] = templatize(value, situation, min_length)
        elif isinstance(value, list):
            result[name] = [templatize(item, situation, min_length) for item in value]
        else:
            result[name] = value
    return result


def signature(step: SkillStep) -> str:
    """Canonical signature of a templated step."""
    return canonical_json(step)


def eligible(step: TraceStep) -> bool:
    """True if ``step`` may be used as mining evidence."""
    return step.verified and step.result.ok and not step.redacted and step.route != "skill"


def attempt_groups(steps: Sequence[TraceStep]) -> dict[tuple[str, str, int], list[TraceStep]]:
    """Eligible steps grouped per task attempt, one record per position, ordered by index."""
    groups: dict[tuple[str, str, int], dict[int, TraceStep]] = defaultdict(dict)
    for step in steps:
        if eligible(step):
            groups[(step.run_id, step.task_id, step.attempt)][step.index] = step
    return {key: [by_index[i] for i in sorted(by_index)] for key, by_index in groups.items()}


def mine(
    steps: Sequence[TraceStep], *, min_repeats: int, min_template_length: int = 3
) -> list[Pattern]:
    """Return candidate patterns (support descending, then key)."""
    groups = attempt_groups(steps)
    templated: dict[str, SkillStep] = {}
    support: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for (run_id, task_id, _), group in groups.items():
        for step in group:
            skill_step = SkillStep(
                tool=step.action.tool,
                step_kind=step.situation.step_kind,
                args=templatize_args(step.action.args, step.situation, min_template_length),
            )
            templated[step.id] = skill_step
            support[signature(skill_step)].add((run_id, task_id))

    def mechanical(step: TraceStep) -> bool:
        return len(support[signature(templated[step.id])]) >= min_repeats

    found: dict[str, list[tuple[list[TraceStep], str]]] = defaultdict(list)

    def close(run: list[TraceStep]) -> None:
        if run:
            task_kind = run[0].situation.task_kind
            key = digest([task_kind, [signature(templated[s.id]) for s in run]])
            found[key].append((run, task_kind))

    for group in groups.values():
        current: list[TraceStep] = []
        for step in group:
            if mechanical(step) and (not current or step.index == current[-1].index + 1):
                current.append(step)
                continue
            close(current)
            current = [step] if mechanical(step) else []
        close(current)

    patterns: list[Pattern] = []
    for key, runs in found.items():
        tasks = {(steps_[0].run_id, steps_[0].task_id) for steps_, _ in runs}
        if len(tasks) < min_repeats:
            continue
        first_run, task_kind = runs[0]
        pattern_steps = [templated[step.id] for step in first_run]
        slot_fields = sorted(
            {
                token_field
                for skill_step in pattern_steps
                for token_field in _slot_fields(skill_step)
            }
        )
        occurrences = sorted(
            (
                Occurrence(
                    run_id=steps_[0].run_id,
                    task_id=steps_[0].task_id,
                    attempt=steps_[0].attempt,
                    start=steps_[0].index,
                    step_ids=[s.id for s in steps_],
                    situation=steps_[0].situation,
                )
                for steps_, _ in runs
            ),
            key=lambda occ: (occ.run_id, occ.task_id, occ.attempt, occ.start),
        )
        patterns.append(
            Pattern(
                key=key,
                task_kind=task_kind,
                steps=pattern_steps,
                slot_fields=slot_fields,
                occurrences=occurrences,
                support=len(tasks),
            )
        )
    patterns.sort(key=lambda p: (-p.support, p.key))
    return patterns


def _slot_fields(step: SkillStep) -> set[str]:
    fields: set[str] = set()
    for value in step.args.values():
        items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        for item in items:
            for token in parse_template(item):
                if isinstance(token, Placeholder):
                    fields.add(token.slot)
    return fields
