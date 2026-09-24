"""Clock, hashing, fault injection, error hierarchy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from crystallizer.clock import FixedClock, SystemClock, iso, make_rng, new_run_id, parse_iso
from crystallizer.errors import (
    BudgetExhaustedError,
    CheckpointError,
    ConfigError,
    CrystallizerError,
    ExitCode,
    HumanApprovalError,
    SimulatedCrash,
    TaskFailedError,
    UsageError,
    WorkspaceLockedError,
)
from crystallizer.faults import NO_FAULTS, FaultInjector
from crystallizer.hashing import canonical_json, digest, sha256_hex, to_jsonable
from crystallizer.schemas import Action


def test_fixed_clock_advances() -> None:
    clock = FixedClock(datetime(2026, 1, 1), timedelta(seconds=2))
    first = clock.now()
    assert first.tzinfo is not None
    assert clock.now() - first == timedelta(seconds=2)
    clock.advance(timedelta(days=1))
    assert clock.now() - first == timedelta(days=1, seconds=4)


def test_system_clock_is_utc() -> None:
    assert SystemClock().now().tzinfo is UTC


def test_iso_roundtrip() -> None:
    moment = datetime(2026, 3, 4, 5, 6, 7, 890, tzinfo=UTC)
    assert parse_iso(iso(moment)) == moment


def test_run_ids_are_deterministic() -> None:
    assert new_run_id(FixedClock(), make_rng(1)) == new_run_id(FixedClock(), make_rng(1))
    assert new_run_id(FixedClock(), make_rng(1)).startswith("run-20260101T000000-")


def test_canonical_json_and_digest() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert canonical_json({"x": "é"}) == '{"x":"é"}'
    assert digest({"a": 1}) == digest({"a": 1})
    assert sha256_hex(b"abc") == sha256_hex("abc")
    assert to_jsonable((Action(tool="t"),)) == [{"tool": "t", "args": {}}]


def test_fault_injector() -> None:
    injector = FaultInjector("runner.after_step", after=2)
    injector.hit("journal.after_begin")
    injector.hit("runner.after_step")
    with pytest.raises(SimulatedCrash):
        injector.hit("runner.after_step")
    NO_FAULTS.hit("runner.after_step")
    with pytest.raises(ValueError, match="unknown fault point"):
        FaultInjector("nope")
    with pytest.raises(ValueError, match="after"):
        FaultInjector("runner.after_step", after=0)


def test_simulated_crash_is_not_an_exception() -> None:
    assert not issubclass(SimulatedCrash, Exception)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (CrystallizerError, ExitCode.GENERAL),
        (ConfigError, ExitCode.USAGE),
        (UsageError, ExitCode.USAGE),
        (TaskFailedError, ExitCode.TASK_FAILED),
        (HumanApprovalError, ExitCode.HUMAN_UNAVAILABLE),
        (WorkspaceLockedError, ExitCode.LOCKED),
        (CheckpointError, ExitCode.CHECKPOINT_UNRECOVERABLE),
        (BudgetExhaustedError, ExitCode.BUDGET_EXHAUSTED),
    ],
)
def test_exit_codes(error: type[CrystallizerError], code: ExitCode) -> None:
    assert error("x").exit_code is code
