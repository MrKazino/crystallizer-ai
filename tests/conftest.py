"""Shared fixtures. Tests are deterministic: fixed clocks, seeded RNGs, no network, no API keys."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from crystallizer.clock import FixedClock
from crystallizer.config import Config, ToolsConfig
from crystallizer.db import Database
from crystallizer.redaction import Redactor, configure_default
from crystallizer.tools import Sandbox

settings.register_profile(
    "deterministic",
    derandomize=True,
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
settings.load_profile("deterministic")


@pytest.fixture(autouse=True)
def _reset_default_redactor() -> Iterator[None]:
    configure_default(Redactor())
    yield
    configure_default(Redactor())


@pytest.fixture
def clock() -> FixedClock:
    """A deterministic clock."""
    return FixedClock()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace directory."""
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def state_dir(workspace: Path) -> Path:
    """The default state directory inside the workspace."""
    path = workspace / ".crystallizer"
    path.mkdir()
    return path


@pytest.fixture
def db(state_dir: Path) -> Iterator[Database]:
    """An open state database, closed after the test."""
    database = Database(state_dir / "state.db")
    yield database
    database.close()


@pytest.fixture
def sandbox(workspace: Path, state_dir: Path) -> Sandbox:
    """A sandbox with default tool limits."""
    return Sandbox(
        workspace, state_dir, ToolsConfig(), Redactor(), environ={"PATH": "/usr/bin:/bin"}
    )


@pytest.fixture
def config() -> Config:
    """Default configuration."""
    return Config()
