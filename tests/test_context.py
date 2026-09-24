"""Context builder: ordering, relevant files, budget invariant, over-budget dropping."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from crystallizer.clock import FixedClock
from crystallizer.context import ContextBuilder, estimate_tokens, render_task
from crystallizer.db import Database
from crystallizer.memory import Memory
from crystallizer.redaction import REDACTED, Redactor
from crystallizer.schemas import MemoryKind, StepSpec, Task
from crystallizer.tools import Sandbox


def make_task(**kwargs: object) -> Task:
    base: dict[str, object] = {
        "id": "add-parser",
        "title": "Add parser module src/parser.py",
        "kind": "add_module",
        "params": {"name": "parser", "path": "src/parser.py"},
        "acceptance_commands": [["python", "-m", "pytest", "tests/test_parser.py"]],
        "steps": [StepSpec(kind="scaffold", description="write docs/notes.md")],
    }
    base.update(kwargs)
    return Task.model_validate(base)


def builder(sandbox: Sandbox, memory: Memory | None, budget: int = 6000) -> ContextBuilder:
    return ContextBuilder(
        sandbox, memory, Redactor(), budget_tokens=budget, chars_per_token=4, memory_items=5
    )


def test_estimator() -> None:
    assert estimate_tokens("", 4) == 0
    assert estimate_tokens("abcd", 4) == 1
    assert estimate_tokens("abcde", 4) == 2


def test_sections_and_relevant_files(sandbox: Sandbox, workspace: Path, db: Database) -> None:
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "docs").mkdir()
    (workspace / "src/parser.py").write_text("def parse(): ...\n", encoding="utf-8")
    (workspace / "tests/test_parser.py").write_text("def test(): ...\n", encoding="utf-8")
    (workspace / "docs/notes.md").write_text("notes", encoding="utf-8")
    (workspace / "docs/memo.md").write_text("memo password=pw1", encoding="utf-8")
    memory = Memory(db, FixedClock(), Redactor())
    memory.add(MemoryKind.DECISION, "parser uses docs/memo.md", source_task="add-parser")
    memory.add(MemoryKind.NOTE, "unrelated", source_task="other")
    result = builder(sandbox, memory).build(make_task(), trace_digest=["file_write ok", "older"])
    sections = [item.section for item in result.items]
    assert sections == sorted(sections, key=["task", "memory", "file", "trace"].index)
    files = [item.key for item in result.items if item.section == "file"]
    assert files == ["docs/notes.md", "src/parser.py", "tests/test_parser.py", "docs/memo.md"]
    memo = next(item for item in result.items if item.key == "docs/memo.md")
    assert "pw1" not in memo.text
    assert REDACTED in memo.text
    traces = [item.key for item in result.items if item.section == "trace"]
    assert traces == ["t0000", "t0001"]
    assert result.dropped == []
    assert result.total_tokens <= result.budget_tokens
    assert estimate_tokens(result.render(), 4) <= result.budget_tokens


def test_missing_and_escaping_files_are_ignored(sandbox: Sandbox) -> None:
    task = make_task(params={"a": "missing/file.py", "b": "../outside.txt"})
    result = builder(sandbox, None).build(task)
    assert [item.section for item in result.items] == ["task"]


def test_over_budget_drops_lowest_scores_first(sandbox: Sandbox, workspace: Path) -> None:
    (workspace / "src").mkdir()
    (workspace / "src/parser.py").write_text("x" * 400, encoding="utf-8")
    task = make_task()
    task_tokens = estimate_tokens(render_task(task) + "\n\n", 4)
    result = builder(sandbox, None, budget=task_tokens + 110).build(
        task, trace_digest=["recent one", "older two"]
    )
    kept = [item.key for item in result.items]
    dropped = [item.key for item in result.dropped]
    assert "src/parser.py" in kept
    assert dropped == ["t0001", "t0000"]
    assert result.total_tokens <= result.budget_tokens


def test_task_is_truncated_when_it_alone_exceeds_budget(sandbox: Sandbox) -> None:
    task = make_task(title="t" * 400)
    result = builder(sandbox, None, budget=20).build(task)
    assert result.truncated == ["add-parser"]
    assert result.items[0].text.endswith("[truncated]")
    assert estimate_tokens(result.render(), 4) <= 20
    tiny = builder(sandbox, None, budget=1).build(task)
    assert tiny.items[0].text == "##"
    assert estimate_tokens(tiny.render(), 4) <= 1
    assert builder(sandbox, None, budget=4).build(task, budget_tokens=4).total_tokens <= 4


def test_memory_items_zero_skips_memory(sandbox: Sandbox, db: Database) -> None:
    memory = Memory(db, FixedClock(), Redactor())
    memory.add(MemoryKind.NOTE, "parser")
    silent = ContextBuilder(
        sandbox, memory, Redactor(), budget_tokens=1000, chars_per_token=4, memory_items=0
    )
    assert all(item.section != "memory" for item in silent.build(make_task()).items)


@given(
    budget=st.integers(1, 400),
    cpt=st.integers(1, 8),
    digest=st.lists(st.text(max_size=80), max_size=6),
    title=st.text(min_size=1, max_size=300),
)
def test_context_never_exceeds_budget(
    sandbox: Sandbox, budget: int, cpt: int, digest: list[str], title: str
) -> None:
    context = ContextBuilder(
        sandbox, None, Redactor(), budget_tokens=budget, chars_per_token=cpt, memory_items=0
    )
    result = context.build(make_task(title=title), trace_digest=digest)
    assert result.total_tokens <= budget
    assert estimate_tokens(result.render(), cpt) <= budget
