"""Token-budgeted context builder.

``build(task)`` returns items in a fixed section order: the task, relevant memory, relevant files,
and a recent trace digest. Within a section items are ordered by score (descending) then key.

Budget invariant: every item costs ``ceil((len(text) + 2) / chars_per_token)`` tokens (the ``+ 2``
pays for the blank-line separator), and the sum never exceeds the budget. Because
``ceil(a + b) <= ceil(a) + ceil(b)``, the rendered context never exceeds the budget either.
When over budget, the lowest-score non-task items are dropped first and recorded. The task item
is mandatory; if it alone exceeds the budget it is truncated (and recorded as truncated).

Relevant files are those named in the task (params, title, step descriptions), in its
acceptance commands, or in memory entries recorded for the task, each capped by
``max_file_bytes``. Every item is redacted before it can reach a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from crystallizer.errors import SandboxError
from crystallizer.memory import Memory
from crystallizer.redaction import Redactor
from crystallizer.schemas import Strict, Task
from crystallizer.tools import Sandbox

Section = Literal["task", "memory", "file", "trace"]
SECTION_ORDER: tuple[Section, ...] = ("task", "memory", "file", "trace")
SEPARATOR = "\n\n"
TRUNCATED = " [truncated]"
TASK_SCORE = 1e9
_PATHLIKE = re.compile(r"[A-Za-z0-9_./\-]+\.[A-Za-z0-9]+")


def estimate_tokens(text: str, chars_per_token: int) -> int:
    """Documented estimator: ``ceil(len(text) / chars_per_token)``."""
    return -(-len(text) // chars_per_token)


class ContextItem(Strict):
    """One context block."""

    section: Section
    key: str
    score: float
    text: str
    tokens: int = Field(ge=0)


class DroppedItem(Strict):
    """An item left out because of the budget."""

    section: Section
    key: str
    score: float
    tokens: int


class ContextResult(Strict):
    """The built context and what was dropped or truncated."""

    items: list[ContextItem]
    dropped: list[DroppedItem] = Field(default_factory=list)
    truncated: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int

    def render(self) -> str:
        """Concatenate the items with blank-line separators."""
        return SEPARATOR.join(item.text for item in self.items)


def render_task(task: Task) -> str:
    """Render a task as a context block."""
    lines = [
        "## Task",
        f"id: {task.id}",
        f"title: {task.title}",
        f"kind: {task.kind}",
        f"params: {json.dumps(task.params, sort_keys=True)}",
    ]
    if task.steps:
        lines.append("steps:")
        lines.extend(f"  - {step.kind}: {step.description}" for step in task.steps)
    if task.acceptance_commands:
        lines.append("acceptance:")
        lines.extend(f"  - {' '.join(command)}" for command in task.acceptance_commands)
    return "\n".join(lines)


class ContextBuilder:
    """Builds budgeted, deterministic, redacted context for a task."""

    def __init__(
        self,
        sandbox: Sandbox,
        memory: Memory | None,
        redactor: Redactor,
        *,
        budget_tokens: int,
        chars_per_token: int,
        memory_items: int = 5,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        """Configure the builder."""
        self._sandbox = sandbox
        self._memory = memory
        self._redactor = redactor
        self._budget = budget_tokens
        self._cpt = chars_per_token
        self._memory_items = memory_items
        self._max_file_bytes = max_file_bytes

    def _item(self, section: Section, key: str, score: float, text: str) -> ContextItem:
        clean = self._redactor.redact(text)
        cost = estimate_tokens(clean + SEPARATOR, self._cpt)
        return ContextItem(section=section, key=key, score=score, text=clean, tokens=cost)

    def _file_candidates(self, task: Task) -> dict[str, float]:
        scores: dict[str, float] = {}

        def offer(raw: str, score: float) -> None:
            for token in _PATHLIKE.findall(raw):
                if score > scores.get(token, 0.0):
                    scores[token] = score

        for value in task.params.values():
            if isinstance(value, str):
                offer(value, 3.0)
        offer(task.title, 3.0)
        for step in task.steps:
            offer(step.description, 3.0)
        for command in task.acceptance_commands:
            for argument in command:
                offer(argument, 2.0)
        if self._memory is not None:
            for entry in self._memory.entries():
                if entry.source_task == task.id:
                    offer(entry.text, 1.0)
        return scores

    def _file_items(self, task: Task) -> list[ContextItem]:
        items: list[ContextItem] = []
        for name, score in sorted(self._file_candidates(task).items()):
            try:
                path = self._sandbox.resolve(name, write=False)
            except SandboxError:
                continue
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                raw = handle.read(self._max_file_bytes)
            content = raw.decode("utf-8", errors="replace")
            items.append(self._item("file", name, score, f"## File {name}\n{content}"))
        return items

    def build(
        self, task: Task, trace_digest: Sequence[str] = (), budget_tokens: int | None = None
    ) -> ContextResult:
        """Build context for ``task`` within ``budget_tokens`` (default: configured budget)."""
        budget = self._budget if budget_tokens is None else budget_tokens
        task_item = self._item("task", task.id, TASK_SCORE, render_task(task))
        truncated: list[str] = []
        if task_item.tokens > budget:
            task_item = self._truncate(task_item, budget)
            truncated.append(task.id)
        candidates: list[ContextItem] = []
        if self._memory is not None and self._memory_items > 0:
            query = f"{task.title} {task.kind} {' '.join(str(v) for v in task.params.values())}"
            for scored in self._memory.query(query, self._memory_items):
                entry = scored.entry
                text = f"## Memory {entry.id} ({entry.kind.value})\n{entry.text}"
                candidates.append(self._item("memory", f"m{entry.id:08d}", scored.score, text))
        candidates.extend(self._file_items(task))
        for position, line in enumerate(trace_digest):
            score = 0.5 / (1 + position)
            candidates.append(self._item("trace", f"t{position:04d}", score, f"## Recent\n{line}"))
        rank = {name: index for index, name in enumerate(SECTION_ORDER)}
        # Highest priority first; items are dropped from the end (lowest score) until it fits.
        by_priority = sorted(candidates, key=lambda i: (-i.score, rank[i.section], i.key))
        base = task_item.tokens if task_item.text else 0
        used = base + sum(item.tokens for item in by_priority)
        dropped: list[DroppedItem] = []
        while by_priority and used > budget:
            item = by_priority.pop()
            used -= item.tokens
            dropped.append(
                DroppedItem(
                    section=item.section, key=item.key, score=item.score, tokens=item.tokens
                )
            )
        kept: list[ContextItem] = ([task_item] if task_item.text else []) + by_priority
        kept.sort(key=lambda i: (rank[i.section], -i.score, i.key))
        return ContextResult(
            items=kept,
            dropped=dropped,
            truncated=truncated,
            total_tokens=used,
            budget_tokens=budget,
        )

    def _truncate(self, item: ContextItem, budget: int) -> ContextItem:
        allowed = budget * self._cpt - len(SEPARATOR)
        if allowed <= 0:
            text = ""
        elif allowed <= len(TRUNCATED):
            text = item.text[:allowed]
        else:
            text = item.text[: allowed - len(TRUNCATED)] + TRUNCATED
        tokens = estimate_tokens(text + SEPARATOR, self._cpt) if text else 0
        return item.model_copy(update={"text": text, "tokens": tokens})
