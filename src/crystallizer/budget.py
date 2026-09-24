"""Per-run budget governor.

Limits are checked *before* every model call, so a run stops as soon as a limit is reached. The
call that crosses a limit is allowed to finish (its cost is unknown until it returns); the next
call is refused with :class:`BudgetExhaustedError` (exit code 7). A value of 0 disables a limit.
"""

from __future__ import annotations

from crystallizer.config import BudgetConfig
from crystallizer.errors import BudgetExhaustedError
from crystallizer.schemas import Usage

__all__ = ["BudgetGovernor", "Usage"]


class BudgetGovernor:
    """Tracks spending for one run and refuses calls once a limit is reached."""

    def __init__(self, config: BudgetConfig, spent: Usage | None = None) -> None:
        """Start from ``spent`` (used when resuming a run)."""
        self._config = config
        self.spent = spent or Usage()

    def check(self) -> None:
        """Raise :class:`BudgetExhaustedError` if any limit has been reached."""
        limits = self._config
        if limits.max_cost_per_run > 0 and self.spent.cost >= limits.max_cost_per_run:
            raise BudgetExhaustedError(
                f"cost budget exhausted: {self.spent.cost:.6f} >= {limits.max_cost_per_run}"
            )
        if limits.max_tokens_per_run > 0 and self.spent.tokens >= limits.max_tokens_per_run:
            raise BudgetExhaustedError(
                f"token budget exhausted: {self.spent.tokens} >= {limits.max_tokens_per_run}"
            )
        if (
            limits.max_model_calls_per_run > 0
            and self.spent.model_calls >= limits.max_model_calls_per_run
        ):
            raise BudgetExhaustedError(
                f"model-call budget exhausted: {self.spent.model_calls} >= "
                f"{limits.max_model_calls_per_run}"
            )

    def charge(self, usage: Usage) -> None:
        """Add ``usage`` to the running total."""
        self.spent = self.spent.plus(usage)
