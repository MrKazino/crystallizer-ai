"""Database, lock, memory, journal and checkpoint behavior, including failure injection."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crystallizer.checkpoint import CheckpointStore, atomic_write, checkpoint_hash
from crystallizer.clock import FixedClock
from crystallizer.db import Database
from crystallizer.errors import (
    CheckpointError,
    ExitCode,
    InDoubtActionError,
    SimulatedCrash,
    WorkspaceLockedError,
)
from crystallizer.faults import FaultInjector
from crystallizer.journal import Journal, action_hash
from crystallizer.lock import RunLock, pid_alive
from crystallizer.memory import Memory, words
from crystallizer.models import MockModel, MockScript
from crystallizer.redaction import REDACTED, Redactor
from crystallizer.schemas import (
    Action,
    Cursor,
    JournalStatus,
    MemoryKind,
    Plan,
    Task,
    TaskState,
    ToolResult,
)

# --------------------------------------------------------------------------- database


def test_database_migrations_are_idempotent(state_dir: Path) -> None:
    with Database(state_dir / "state.db") as first:
        assert first.schema_version == 1
    with Database(state_dir / "state.db") as second:
        assert second.schema_version == 1
        rows = second.query("PRAGMA journal_mode")
        assert rows[0][0] == "wal"


def test_transaction_rolls_back(db: Database) -> None:
    def insert_then_fail() -> None:
        with db.transaction():
            db.execute(
                "INSERT INTO task_events (run_id, task_id, from_state, to_state, reason, ts)"
                " VALUES ('r', 't', 'a', 'b', 'x', 'ts')"
            )
            with db.transaction():
                pass
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        insert_then_fail()
    assert db.query("SELECT COUNT(*) AS c FROM task_events")[0]["c"] == 0


# --------------------------------------------------------------------------- lock


def test_second_concurrent_run_fails_with_exit_code_5(state_dir: Path) -> None:
    with RunLock(state_dir):
        other = RunLock(state_dir, pid=999_999_999, is_alive=lambda _pid: True)
        with pytest.raises(WorkspaceLockedError) as info:
            other.acquire()
        assert info.value.exit_code is ExitCode.LOCKED
    assert not (state_dir / "run.lock").exists()


def test_stale_lock_is_recovered(state_dir: Path) -> None:
    (state_dir / "run.lock").write_text("424242", encoding="utf-8")
    lock = RunLock(state_dir, is_alive=lambda _pid: False)
    lock.acquire()
    assert (state_dir / "run.lock").read_text(encoding="utf-8") == str(lock.pid)
    lock.release()
    lock.release()
    assert not (state_dir / "run.lock").exists()


def test_garbage_lock_is_stale(state_dir: Path) -> None:
    (state_dir / "run.lock").write_text("not-a-pid", encoding="utf-8")
    with RunLock(state_dir) as lock:
        assert lock.held


def test_lock_contention_that_never_clears(state_dir: Path) -> None:
    class Contended(RunLock):
        def _remove_stale(self) -> None:
            """Another process re-creates the lock as soon as it is removed."""

    (state_dir / "run.lock").write_text("1", encoding="utf-8")
    with pytest.raises(WorkspaceLockedError):
        Contended(state_dir, is_alive=lambda _pid: False).acquire()


def test_lock_released_on_crash(state_dir: Path) -> None:
    with pytest.raises(SimulatedCrash), RunLock(state_dir):
        raise SimulatedCrash("crash")
    assert not (state_dir / "run.lock").exists()


def test_pid_alive() -> None:
    import os

    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(2**22 + 12345)


# --------------------------------------------------------------------------- memory


def test_memory_add_query_is_deterministic(db: Database, clock: FixedClock) -> None:
    memory = Memory(db, clock, Redactor())
    first = memory.add(MemoryKind.DECISION, "use sqlite for parser state", ["arch"], "t1")
    memory.add(MemoryKind.NOTE, "parser needs tests")
    memory.add(MemoryKind.FACT, "unrelated fact")
    results = memory.query("parser state", k=2)
    assert [r.entry.id for r in results] == [first.id, 2]
    assert results[0].score > results[1].score
    assert memory.query("parser", k=0) == []
    assert memory.max_id() == 3
    assert memory.get(first.id).tags == ["arch"]
    with pytest.raises(KeyError):
        memory.get(99)


def test_memory_recency_breaks_overlap_ties(db: Database) -> None:
    clock = FixedClock(step=timedelta(days=10))
    memory = Memory(db, clock, Redactor(), half_life_days=10)
    old = memory.add(MemoryKind.NOTE, "alpha")
    new = memory.add(MemoryKind.NOTE, "alpha")
    assert [r.entry.id for r in memory.query("alpha", 2)] == [new.id, old.id]
    assert [r.entry.id for r in memory.query("", 2)] == [new.id, old.id]


def test_memory_redacts_text(db: Database, clock: FixedClock) -> None:
    memory = Memory(db, clock, Redactor())
    entry = memory.add(MemoryKind.NOTE, "the api_key=abc123 was rotated")
    assert "abc123" not in entry.text
    assert REDACTED in entry.text


def test_memory_summarize_archives_never_deletes(db: Database) -> None:
    clock = FixedClock(step=timedelta(days=1))
    memory = Memory(db, clock, Redactor())
    for index in range(3):
        memory.add(MemoryKind.NOTE, f"note {index}")
    cutoff = clock.now()
    memory.add(MemoryKind.NOTE, "recent")
    summary, completion = memory.summarize(cutoff, MockModel(MockScript()))
    assert summary is not None
    assert completion is not None
    assert summary.summary_of == [1, 2, 3]
    assert "(3 entries)" in summary.text
    assert [e.id for e in memory.entries()] == [4, summary.id]
    assert len(memory.entries(include_archived=True)) == 5
    assert memory.summarize(FixedClock().now() - timedelta(days=365), MockModel(MockScript())) == (
        None,
        None,
    )


def test_words() -> None:
    assert words("Hello, World-42!") == {"hello", "world", "42"}


# --------------------------------------------------------------------------- journal


def ok_result(text: str = "done") -> ToolResult:
    return ToolResult(ok=True, exit_code=0, output=text)


def test_journal_executes_once_and_replays(db: Database, clock: FixedClock) -> None:
    journal = Journal(db, clock)
    calls: list[int] = []
    action = Action(tool="file_write", args={"path": "a", "content": "b"})

    def run() -> ToolResult:
        calls.append(1)
        return ok_result()

    kwargs = {"run_id": "r", "task_id": "t", "attempt": 1, "step_index": 0, "action": action}
    first = journal.execute(**kwargs, run=run, allow_redo=True)  # type: ignore[arg-type]
    second = journal.execute(**kwargs, run=run, allow_redo=True)  # type: ignore[arg-type]
    assert calls == [1]
    assert not first.replayed
    assert second.replayed
    assert second.result == first.result
    third = journal.execute(  # a retry is a new attempt and runs again
        run_id="r", task_id="t", attempt=2, step_index=0, action=action, run=run, allow_redo=True
    )
    assert calls == [1, 1]
    assert not third.replayed
    entries = journal.entries("r")
    assert [e.attempt for e in entries] == [1, 2]
    assert all(e.status is JournalStatus.COMPLETED for e in entries)
    assert journal.entries()[0].action_hash == action_hash(action)


@pytest.mark.parametrize("point", ["journal.after_begin", "journal.after_execute"])
def test_journal_in_doubt_after_crash(db: Database, clock: FixedClock, point: str) -> None:
    action = Action(tool="file_delete", args={"path": "a"})
    crashing = Journal(db, clock, FaultInjector(point))
    kwargs = {"run_id": "r", "task_id": "t", "attempt": 1, "step_index": 0, "action": action}
    with pytest.raises(SimulatedCrash):
        crashing.execute(**kwargs, run=ok_result, allow_redo=False)  # type: ignore[arg-type]
    journal = Journal(db, clock)
    key = Journal.key("r", "t", 1, 0, action)
    assert journal.lookup(key) is not None
    with pytest.raises(InDoubtActionError):
        journal.execute(**kwargs, run=ok_result, allow_redo=False)  # type: ignore[arg-type]
    outcome = journal.execute(**kwargs, run=ok_result, allow_redo=True)  # type: ignore[arg-type]
    assert outcome.redone
    assert journal.lookup(key) is not None


def test_journal_completed_without_result_is_in_doubt(db: Database, clock: FixedClock) -> None:
    journal = Journal(db, clock)
    action = Action(tool="run_tests")
    key = Journal.key("r", "t", 1, 0, action)
    db.execute(
        "INSERT INTO journal VALUES (?, 'r', 't', 1, 0, 'h', 'completed', NULL, 'ts')", (key,)
    )
    with pytest.raises(InDoubtActionError):
        journal.execute(
            run_id="r",
            task_id="t",
            attempt=1,
            step_index=0,
            action=action,
            run=ok_result,
            allow_redo=True,
        )


# --------------------------------------------------------------------------- checkpoints


def make_plan(n: int = 2) -> Plan:
    return Plan(
        goal="build it",
        tasks=[Task(id=f"t{i}", title=f"Task {i}", kind="k", params={"n": i}) for i in range(n)],
    )


def test_checkpoint_save_load_and_prune(state_dir: Path, clock: FixedClock) -> None:
    store = CheckpointStore(state_dir, clock, Redactor())
    assert store.latest() is None
    assert not store.exists()
    for index in range(5):
        store.save(make_plan(), Cursor(run_id="r", active=True, attempt=index), index, 0)
    files = sorted(p.name for p in store.directory.iterdir())
    assert files == ["ckpt-00000003.json", "ckpt-00000004.json", "ckpt-00000005.json"]
    latest = store.latest()
    assert latest is not None
    assert latest.seq == 5
    assert latest.cursor.attempt == 4
    assert latest.hash == checkpoint_hash(latest)


def test_corrupt_latest_falls_back(state_dir: Path, clock: FixedClock) -> None:
    store = CheckpointStore(state_dir, clock, Redactor())
    store.save(make_plan(), Cursor(), 1, 0)
    second = store.save(make_plan(3), Cursor(), 2, 0)
    path = store.directory / f"ckpt-{second.seq:08d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["memory_snapshot_id"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    latest = store.latest()
    assert latest is not None
    assert latest.seq == 1


def test_no_valid_checkpoint_is_exit_6(state_dir: Path, clock: FixedClock) -> None:
    store = CheckpointStore(state_dir, clock, Redactor())
    store.save(make_plan(), Cursor(), 1, 0)
    (store.directory / "ckpt-00000001.json").write_text("{garbage", encoding="utf-8")
    with pytest.raises(CheckpointError) as info:
        store.latest()
    assert info.value.exit_code is ExitCode.CHECKPOINT_UNRECOVERABLE


def test_checkpoint_redacts_plan(state_dir: Path, clock: FixedClock) -> None:
    store = CheckpointStore(state_dir, clock, Redactor())
    plan = Plan(goal="deploy with token=abc123", tasks=[Task(id="t", title="x", kind="k")])
    saved = store.save(plan, Cursor(), 0, 0)
    assert "abc123" not in saved.plan.goal
    raw = (store.directory / "ckpt-00000001.json").read_text(encoding="utf-8")
    assert "abc123" not in raw


def test_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.json"
    atomic_write(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert [p.name for p in target.parent.iterdir()] == ["file.json"]


_params = st.dictionaries(
    st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True),
    st.one_of(st.integers(-1000, 1000), st.booleans(), st.text(max_size=12)),
    max_size=4,
)


@given(
    params=_params,
    state=st.sampled_from(list(TaskState)),
    attempt=st.integers(0, 50),
    snapshot=st.integers(0, 10_000),
)
def test_checkpoint_roundtrip_property(
    tmp_path_factory: pytest.TempPathFactory,
    params: dict[str, str | int | float | bool],
    state: TaskState,
    attempt: int,
    snapshot: int,
) -> None:
    state_dir = tmp_path_factory.mktemp("ckpt")
    redactor = Redactor()
    params = redactor.redact_obj(params)
    store = CheckpointStore(state_dir, FixedClock(), redactor)
    plan = Plan(goal="g", tasks=[Task(id="a", title="A", kind="k", params=params, state=state)])
    saved = store.save(plan, Cursor(run_id="r", attempt=attempt, active=True), snapshot, 3)
    loaded = store.latest()
    assert loaded == saved
