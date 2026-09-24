"""Exception hierarchy and the process exit-code contract.

Every error the harness raises on purpose derives from :class:`CrystallizerError` and carries the
exit code the CLI must return for it. Messages are redacted at construction time so an error can
never carry a secret into a log line, a trace, or the terminal.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from crystallizer.redaction import default_redactor

if TYPE_CHECKING:
    from crystallizer.schemas import Usage


class ExitCode(IntEnum):
    """Process exit codes. Documented in README; stable across versions."""

    OK = 0
    GENERAL = 1
    USAGE = 2
    TASK_FAILED = 3
    HUMAN_UNAVAILABLE = 4
    LOCKED = 5
    CHECKPOINT_UNRECOVERABLE = 6
    BUDGET_EXHAUSTED = 7


class CrystallizerError(Exception):
    """Base class for all intentional harness errors."""

    exit_code: ExitCode = ExitCode.GENERAL

    def __init__(self, message: str) -> None:
        """Store a redacted copy of ``message``.

        ``usage`` may be attached by the router when model calls were made before the error
        (for example a budget stop mid-step), so the runner can still account for them.
        """
        self.message = default_redactor().redact(message)
        self.usage: Usage | None = None
        super().__init__(self.message)


class ConfigError(CrystallizerError):
    """Invalid configuration file, unknown key, or unknown profile."""

    exit_code = ExitCode.USAGE


class UsageError(CrystallizerError):
    """The command line or API was used incorrectly."""

    exit_code = ExitCode.USAGE


class TaskFailedError(CrystallizerError):
    """At least one task failed or was blocked."""

    exit_code = ExitCode.TASK_FAILED


class HumanApprovalError(CrystallizerError):
    """Human approval was required but is unavailable (no TTY) or was denied."""

    exit_code = ExitCode.HUMAN_UNAVAILABLE


class WorkspaceLockedError(CrystallizerError):
    """Another live run holds the workspace lock."""

    exit_code = ExitCode.LOCKED


class CheckpointError(CrystallizerError):
    """No valid checkpoint could be restored."""

    exit_code = ExitCode.CHECKPOINT_UNRECOVERABLE


class BudgetExhaustedError(CrystallizerError):
    """A configured per-run budget limit was reached."""

    exit_code = ExitCode.BUDGET_EXHAUSTED


class SandboxError(CrystallizerError):
    """A tool call violated the sandbox (path escape, protected path, disallowed executable)."""


class ToolError(CrystallizerError):
    """A tool call was malformed or referenced an unknown tool."""


class PlanError(CrystallizerError):
    """A plan failed validation (bad JSON, unknown dependency, cycle)."""


class SkillError(CrystallizerError):
    """A skill failed validation or could not be instantiated."""


class ModelError(CrystallizerError):
    """A model provider call failed."""


class ExtensionError(CrystallizerError):
    """A plugin or registry operation was invalid (duplicate name, unknown plugin)."""


class InDoubtActionError(CrystallizerError):
    """A journaled action started before a crash and its outcome is unknown."""


class SimulatedCrash(BaseException):
    """Raised by the fault injector to simulate a process crash in tests.

    Derives from ``BaseException`` so that no ``except Exception`` handler can swallow it.
    """
