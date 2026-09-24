"""Human approval: the last rung of the ladder.

The human either approves or denies a pending proposal, or (when no tier produced one) types an
action as JSON. Interactive approval requires a TTY; without one the harness fails closed with
exit code 4. Denial is also exit code 4.
"""

from __future__ import annotations

import json
import sys
from typing import Protocol, TextIO

from pydantic import ValidationError

from crystallizer.errors import HumanApprovalError
from crystallizer.schemas import Action, ApprovalRequest


class Approver(Protocol):
    """Human decision maker."""

    def approve(self, request: ApprovalRequest) -> bool:
        """Return True to approve the request's actions."""
        ...

    def propose(self, request: ApprovalRequest) -> Action | None:
        """Return an action for a step no tier could answer, or None to give up."""
        ...


def describe(request: ApprovalRequest) -> str:
    """Human-readable summary of an approval request."""
    lines = [
        f"Approval needed: {request.reason}",
        f"  run {request.run_id}, task {request.task_id}, step {request.step_index} "
        f"({request.situation.step_kind})",
    ]
    for action, decision in zip(request.actions, request.decisions, strict=False):
        lines.append(f"  - {action.tool} {json.dumps(action.args, sort_keys=True)}")
        lines.append(f"    {decision.classification.value}: {decision.reason}")
    return "\n".join(lines)


class TTYApprover:
    """Asks on the terminal; fails closed without a TTY."""

    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Use the given streams (defaults: ``sys.stdin`` and ``sys.stderr``)."""
        self._in = stdin if stdin is not None else sys.stdin
        self._out = stdout if stdout is not None else sys.stderr

    def _require_tty(self) -> None:
        if not self._in.isatty():
            raise HumanApprovalError("human approval is required but stdin is not a TTY")

    def approve(self, request: ApprovalRequest) -> bool:
        """Ask ``[y/N]``."""
        self._require_tty()
        self._out.write(describe(request) + "\nApprove? [y/N] ")
        self._out.flush()
        return self._in.readline().strip().lower() in {"y", "yes"}

    def propose(self, request: ApprovalRequest) -> Action | None:
        """Read one action as JSON ``{"tool": ..., "args": {...}}``; blank gives up."""
        self._require_tty()
        self._out.write(describe(request) + "\nEnter an action as JSON (blank to abort): ")
        self._out.flush()
        line = self._in.readline().strip()
        if not line:
            return None
        try:
            return Action.model_validate_json(line)
        except ValidationError:
            self._out.write("not a valid action\n")
            return None


class DenyApprover:
    """Non-interactive approver that always fails closed."""

    def approve(self, request: ApprovalRequest) -> bool:
        """Always raise: no human is available."""
        raise HumanApprovalError(f"human approval unavailable: {request.reason}")

    def propose(self, request: ApprovalRequest) -> Action | None:
        """Always raise: no human is available."""
        raise HumanApprovalError(f"human input unavailable: {request.reason}")


class ScriptedApprover:
    """Deterministic approver for tests: pops scripted answers and records every request."""

    def __init__(
        self, answers: list[bool] | None = None, actions: list[Action | None] | None = None
    ) -> None:
        """Script the answers."""
        self.answers = list(answers or [])
        self.actions = list(actions or [])
        self.requests: list[ApprovalRequest] = []

    def approve(self, request: ApprovalRequest) -> bool:
        """Return the next scripted answer (deny when exhausted)."""
        self.requests.append(request)
        return self.answers.pop(0) if self.answers else False

    def propose(self, request: ApprovalRequest) -> Action | None:
        """Return the next scripted action (None when exhausted)."""
        self.requests.append(request)
        return self.actions.pop(0) if self.actions else None
