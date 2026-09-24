"""The cost ladder: route every step down the cheapest safe path.

Tiers are tried from the attempt's *floor* upward (default ladder: skill, small, large; the human
tier is always last and outside the floors). Each tier either proposes or escalates. A proposal
is then gated by policy:

* a skill that emits an irreversible action is unsafe: it is demoted and the step escalates;
* an irreversible action that is not allow-listed goes straight to the human tier;
* an allow-listed irreversible action needs confidence >= ``irreversible_confidence``,
  otherwise the next tier is asked (the proposal stays pending for the human).

When no tier accepts, the human approves the pending proposal or supplies an action; without a
human tier the step fails (or, if human approval was required, the run stops with exit 4).
Usage spent before an interruption is attached to the error so the runner can record it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from crystallizer.errors import BudgetExhaustedError, HumanApprovalError
from crystallizer.logging_setup import get_logger
from crystallizer.policy import Policy
from crystallizer.schemas import (
    EscalationReason,
    Proposal,
    RouteDecision,
    StepRequest,
    TierResult,
    Usage,
)
from crystallizer.tiers import HumanTier, Proposer

_log = get_logger("router")


class NoProposal(Exception):  # noqa: N818 - a routing outcome, not an error for users
    """No tier produced an acceptable proposal; carries the usage spent trying."""

    def __init__(self, usage: Usage, path: list[str]) -> None:
        """Record usage and the escalation path."""
        super().__init__(" -> ".join(path) or "no proposal")
        self.usage = usage
        self.path = path


class StepRouter(Protocol):
    """Chooses who answers a step."""

    @property
    def max_floor(self) -> int:
        """Highest floor index an attempt may start from before the task fails."""
        ...

    def route(self, request: StepRequest, floor: int) -> RouteDecision:
        """Return an accepted proposal or raise :class:`NoProposal`."""
        ...


class LadderRouter:
    """Routes steps through the configured tiers, then the human."""

    def __init__(
        self,
        tiers: Sequence[Proposer],
        *,
        human: HumanTier | None,
        policy: Policy,
        irreversible_confidence: float,
        on_unsafe_skill: Callable[[str, str], None] | None = None,
    ) -> None:
        """Configure the ladder (``tiers`` excludes the human tier)."""
        if not tiers:
            raise ValueError("the ladder needs at least one non-human tier")
        self._tiers = list(tiers)
        self._human = human
        self._policy = policy
        self._threshold = irreversible_confidence
        self._on_unsafe_skill = on_unsafe_skill

    @property
    def max_floor(self) -> int:
        """Index of the last non-human tier."""
        return len(self._tiers) - 1

    @property
    def tier_names(self) -> list[str]:
        """Names of the non-human tiers in order."""
        return [tier.name for tier in self._tiers]

    def _ask(self, tier: Proposer, request: StepRequest, spent: Usage) -> TierResult:
        try:
            return tier.propose(request)
        except (BudgetExhaustedError, HumanApprovalError) as exc:
            exc.usage = spent.plus(exc.usage or Usage())
            raise
        except Exception as exc:  # noqa: BLE001 - a faulty tier escalates, it never crashes a run
            _log.warning("tier raised", extra={"tier": tier.name, "error": repr(exc)})
            return TierResult(
                tier=tier.name, escalation=EscalationReason.TIER_ERROR, detail=repr(exc)
            )

    def route(self, request: StepRequest, floor: int) -> RouteDecision:
        """Walk the ladder from ``floor`` and return the accepted proposal."""
        path: list[str] = []
        usage = Usage()
        pending: Proposal | None = None
        needs_human = False
        for tier in self._tiers[floor:]:
            result = self._ask(tier, request, usage)
            usage = usage.plus(result.usage)
            if result.proposal is None:
                reason = result.escalation or EscalationReason.TIER_ERROR
                path.append(f"{tier.name}:{reason.value}")
                continue
            proposal = result.proposal
            decisions = [self._policy.classify(action) for action in proposal.actions]
            irreversible = any(decision.irreversible for decision in decisions)
            if tier.name == "skill" and irreversible:
                path.append(f"skill:{EscalationReason.SKILL_UNSAFE.value}")
                if self._on_unsafe_skill is not None and proposal.skill_key is not None:
                    self._on_unsafe_skill(
                        proposal.skill_key, "skill emitted an irreversible action"
                    )
                continue
            if any(decision.requires_human for decision in decisions):
                path.append(f"{tier.name}:{EscalationReason.IRREVERSIBLE_REQUIRES_HUMAN.value}")
                pending = proposal
                needs_human = True
                break
            if irreversible and proposal.confidence < self._threshold:
                path.append(f"{tier.name}:{EscalationReason.IRREVERSIBLE_LOW_CONFIDENCE.value}")
                pending = proposal
                continue
            return RouteDecision(
                route=tier.name,
                proposal=proposal,
                usage=usage,
                escalation_reason=path[-1].split(":", 1)[1] if path else None,
                escalation_path=path,
            )
        return self._escalate_to_human(request, path, usage, pending, needs_human)

    def _escalate_to_human(
        self,
        request: StepRequest,
        path: list[str],
        usage: Usage,
        pending: Proposal | None,
        needs_human: bool,
    ) -> RouteDecision:
        reason = path[-1] if path else "no tier could answer"
        if self._human is None:
            if needs_human or pending is not None:
                error = HumanApprovalError(
                    f"human approval required ({reason}) but the ladder has no human tier"
                )
                error.usage = usage
                raise error
            raise NoProposal(usage, path)
        try:
            proposal = self._human.decide(request, pending, reason)
        except HumanApprovalError as exc:
            exc.usage = usage
            raise
        return RouteDecision(
            route=self._human.name,
            proposal=proposal,
            usage=usage,
            escalation_reason=reason.split(":", 1)[1] if ":" in reason else reason,
            escalation_path=path,
            human_approved=True,
        )
