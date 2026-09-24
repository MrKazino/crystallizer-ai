"""Tiers and the ladder router: voting, escalation reasons, gates, human tier, budgets."""

from __future__ import annotations

from typing import Any

import pytest

from crystallizer.approval import DenyApprover, ScriptedApprover
from crystallizer.budget import BudgetGovernor
from crystallizer.config import BudgetConfig, CostConfig, PolicyConfig
from crystallizer.errors import BudgetExhaustedError, HumanApprovalError
from crystallizer.models import CostTable, MockBehavior, MockModel, MockScript, MockStep
from crystallizer.policy import Policy
from crystallizer.router import LadderRouter, NoProposal
from crystallizer.schemas import (
    Action,
    EscalationReason,
    Proposal,
    Situation,
    Skill,
    SkillManifest,
    StepRequest,
    Task,
    TierResult,
    Usage,
)
from crystallizer.skills.executor import parse_skill
from crystallizer.tiers import HumanTier, ModelTier, SkillTier, find_skill

WRITE = Action(tool="file_write", args={"path": "out/a.txt", "content": "a"})
DELETE = Action(tool="file_delete", args={"path": "out/a.txt"})


def request(kind: str = "write", open_mode: bool = False) -> StepRequest:
    task = Task(
        id="t1",
        title="T",
        kind="greet",
        params={"name": "alpha"},
        steps=[] if open_mode else [{"kind": kind}, {"kind": "read"}],  # type: ignore[list-item]
    )
    return StepRequest(
        run_id="r",
        task=task,
        situation=Situation(
            task_kind="greet", step_kind="open" if open_mode else kind, params={"name": "alpha"}
        ),
        attempt=1,
        planned_kinds=[] if open_mode else [kind, "read"],
        open_mode=open_mode,
        tools=["file_write"],
    )


def model_tier(
    behavior: MockBehavior = MockBehavior.CORRECT,
    *,
    action: Action = WRITE,
    tier: str = "small",
    samples: int = 3,
    budget: BudgetGovernor | None = None,
) -> ModelTier:
    step = MockStep.model_validate({"action": action, tier: behavior})
    return ModelTier(
        tier,
        tier,
        MockModel(MockScript(actions={"t1": [step]})),
        CostTable(CostConfig()),
        budget or BudgetGovernor(BudgetConfig()),
        samples=samples,
        agreement=0.67,
        max_tokens=100,
    )


# --------------------------------------------------------------------------- model tier


def test_model_tier_agreement_and_usage() -> None:
    result = model_tier().propose(request())
    assert result.proposal is not None
    assert result.proposal.actions == [WRITE]
    assert result.proposal.confidence == 1.0
    assert result.usage.model_calls == 3
    assert result.usage.cost == pytest.approx(3 * (100 * 0.25 + 50 * 1.25) / 1e6)


@pytest.mark.parametrize(
    ("behavior", "reason"),
    [
        (MockBehavior.DISAGREE, EscalationReason.DISAGREEMENT),
        (MockBehavior.GARBAGE, EscalationReason.PARSE_ERROR),
        (MockBehavior.ERROR, EscalationReason.TIER_ERROR),
    ],
)
def test_model_tier_escalations(behavior: MockBehavior, reason: EscalationReason) -> None:
    result = model_tier(behavior).propose(request())
    assert result.proposal is None
    assert result.escalation is reason


def test_single_sample_confidence_is_capped() -> None:
    result = model_tier(tier="large", samples=1).propose(request())
    assert result.proposal is not None
    assert result.proposal.confidence == 0.5


def test_budget_exhaustion_carries_partial_usage() -> None:
    budget = BudgetGovernor(BudgetConfig(max_model_calls_per_run=2))
    with pytest.raises(BudgetExhaustedError) as info:
        model_tier(budget=budget).propose(request())
    assert info.value.usage is not None
    assert info.value.usage.model_calls == 2


# --------------------------------------------------------------------------- skill tier


def skill(steps: list[dict[str, Any]], guard_value: str = "greet") -> Skill:
    return parse_skill(
        {
            "id": "skill-aaaaaaaaaaaa",
            "version": 1,
            "task_kind": "greet",
            "slots": {"name": {"type": "identifier", "source": "params.name"}},
            "guard": {"all": [{"field": "task_kind", "op": "eq", "value": guard_value}]},
            "steps": steps,
        }
    )


def manifest(item: Skill) -> SkillManifest:
    return SkillManifest(
        id=item.id,
        version=item.version,
        content_hash="h",
        guard_summary="g",
        source_file="f",
        created_at="t",
        status="active",  # type: ignore[arg-type]
    )


WRITE_STEP = {
    "tool": "file_write",
    "step_kind": "write",
    "args": {"path": "out/{name}.txt", "content": "{name}"},
}
READ_STEP = {"tool": "file_read", "step_kind": "read", "args": {"path": "out/{name}.txt"}}


def test_skill_tier_match_and_reasons() -> None:
    good = skill([WRITE_STEP, READ_STEP])
    versions = [1]
    calls: list[int] = []

    def source() -> list[tuple[Skill, SkillManifest]]:
        calls.append(1)
        return [(good, manifest(good))]

    tier = SkillTier(source, lambda: versions[0])
    result = tier.propose(request())
    assert result.proposal is not None
    assert [a.tool for a in result.proposal.actions] == ["file_write", "file_read"]
    assert result.proposal.skill_key == "skill-aaaaaaaaaaaa@v1"
    tier.propose(request())
    assert calls == [1]  # cached until the registry version changes
    versions[0] = 2
    tier.propose(request())
    assert calls == [1, 1]
    assert tier.propose(request(kind="other")).escalation is EscalationReason.NO_SKILL
    guarded = skill([WRITE_STEP, READ_STEP], guard_value="nope")
    guarded_tier = SkillTier(lambda: [(guarded, manifest(guarded))], lambda: 1)
    assert guarded_tier.propose(request()).escalation is EscalationReason.GUARD_FALSE
    empty = SkillTier(lambda: [], lambda: 1)
    assert empty.propose(request()).escalation is EscalationReason.NO_SKILL


def test_find_skill_open_mode_and_bad_slots() -> None:
    open_skill = skill([{**READ_STEP, "step_kind": "open"}])
    match, applicable = find_skill(
        [(open_skill, manifest(open_skill))], request(open_mode=True).situation, [], True
    )
    assert applicable
    assert match is not None
    planned_skill = skill([WRITE_STEP])
    match, applicable = find_skill(
        [(planned_skill, manifest(planned_skill))], request(open_mode=True).situation, [], True
    )
    assert match is None
    assert not applicable
    no_slot = Situation(task_kind="greet", step_kind="write", params={})
    match, applicable = find_skill(
        [(planned_skill, manifest(planned_skill))], no_slot, ["write"], False
    )
    assert match is None
    assert applicable


# --------------------------------------------------------------------------- router


class StaticTier:
    def __init__(self, name: str, result: TierResult | Exception) -> None:
        self._name = name
        self._result = result
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def propose(self, request: StepRequest) -> TierResult:
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def proposes(
    name: str, action: Action, confidence: float = 1.0, skill_key: str | None = None
) -> StaticTier:
    return StaticTier(
        name,
        TierResult(
            tier=name,
            proposal=Proposal(
                tier=name, actions=[action], confidence=confidence, skill_key=skill_key
            ),
            usage=Usage(model_calls=1, cost=0.1),
        ),
    )


def escalates(name: str, reason: EscalationReason = EscalationReason.DISAGREEMENT) -> StaticTier:
    return StaticTier(
        name, TierResult(tier=name, escalation=reason, usage=Usage(model_calls=3, cost=0.3))
    )


def router(
    tiers: list[StaticTier],
    approver: Any = None,
    allow: list[str] | None = None,
    unsafe: list[str] | None = None,
) -> LadderRouter:
    policy = Policy(PolicyConfig(allow_irreversible=allow or []))
    human = HumanTier(approver, policy) if approver is not None else None
    return LadderRouter(
        tiers,
        human=human,
        policy=policy,
        irreversible_confidence=0.9,
        on_unsafe_skill=(lambda key, _reason: unsafe.append(key)) if unsafe is not None else None,
    )


def test_escalation_path_and_usage_accumulate() -> None:
    ladder = router(
        [
            escalates("skill", EscalationReason.NO_SKILL),
            escalates("small"),
            proposes("large", WRITE),
        ]
    )
    decision = ladder.route(request(), 0)
    assert decision.route == "large"
    assert decision.escalation_path == ["skill:no_skill", "small:disagreement"]
    assert decision.escalation_reason == "disagreement"
    assert decision.usage.model_calls == 7
    assert ladder.max_floor == 2
    assert ladder.tier_names == ["skill", "small", "large"]


def test_floor_skips_lower_tiers() -> None:
    skill_tier = proposes("skill", WRITE)
    ladder = router([skill_tier, proposes("small", WRITE)])
    assert ladder.route(request(), 1).route == "small"
    assert skill_tier.calls == 0


def test_irreversible_goes_to_human() -> None:
    approver = ScriptedApprover(answers=[True])
    ladder = router([proposes("small", DELETE), proposes("large", WRITE)], approver)
    decision = ladder.route(request(), 0)
    assert decision.route == "human"
    assert decision.human_approved
    assert decision.proposal.actions == [DELETE]
    assert decision.escalation_path == ["small:irreversible_requires_human"]
    denied = router([proposes("small", DELETE)], ScriptedApprover(answers=[False]))
    with pytest.raises(HumanApprovalError) as info:
        denied.route(request(), 0)
    assert info.value.usage is not None
    assert info.value.usage.model_calls == 1
    with pytest.raises(HumanApprovalError):
        router([proposes("small", DELETE)], DenyApprover()).route(request(), 0)


def test_allow_listed_irreversible_needs_confidence() -> None:
    confident = router([proposes("small", DELETE, confidence=1.0)], allow=["file_delete"])
    assert confident.route(request(), 0).route == "small"
    approver = ScriptedApprover(answers=[True])
    unsure = router(
        [proposes("small", DELETE, confidence=0.67), proposes("large", DELETE, confidence=0.5)],
        approver,
        allow=["file_delete"],
    )
    decision = unsure.route(request(), 0)
    assert decision.route == "human"
    assert decision.escalation_path == [
        "small:irreversible_low_confidence",
        "large:irreversible_low_confidence",
    ]
    no_human = router([proposes("small", DELETE, confidence=0.5)], allow=["file_delete"])
    with pytest.raises(HumanApprovalError, match="no human tier"):
        no_human.route(request(), 0)


def test_unsafe_skill_is_reported_and_skipped() -> None:
    unsafe: list[str] = []
    ladder = router(
        [proposes("skill", DELETE, skill_key="skill-x@v1"), proposes("small", WRITE)], unsafe=unsafe
    )
    decision = ladder.route(request(), 0)
    assert decision.route == "small"
    assert decision.escalation_path == ["skill:skill_unsafe"]
    assert unsafe == ["skill-x@v1"]


def test_no_human_tier_means_no_proposal() -> None:
    ladder = router([escalates("small"), escalates("large")])
    with pytest.raises(NoProposal) as info:
        ladder.route(request(), 0)
    assert info.value.path == ["small:disagreement", "large:disagreement"]
    assert info.value.usage.model_calls == 6


def test_faulty_tier_escalates() -> None:
    ladder = router([StaticTier("plugin", RuntimeError("boom")), proposes("large", WRITE)])
    decision = ladder.route(request(), 0)
    assert decision.escalation_path == ["plugin:tier_error"]


def test_budget_error_gets_accumulated_usage() -> None:
    ladder = router([escalates("small"), StaticTier("large", BudgetExhaustedError("stop"))])
    with pytest.raises(BudgetExhaustedError) as info:
        ladder.route(request(), 0)
    assert info.value.usage is not None
    assert info.value.usage.model_calls == 3


def test_human_supplies_action_when_nobody_answers() -> None:
    approver = ScriptedApprover(actions=[Action(tool="file_read", args={"path": "./out/a.txt"})])
    ladder = router([escalates("small")], approver)
    decision = ladder.route(request(), 0)
    assert decision.route == "human"
    assert decision.proposal.actions == [Action(tool="file_read", args={"path": "out/a.txt"})]
    with pytest.raises(HumanApprovalError, match="no action"):
        router([escalates("small")], ScriptedApprover()).route(request(), 0)
    tier = HumanTier(ScriptedApprover(actions=[WRITE]), Policy(PolicyConfig()))
    assert tier.propose(request()).proposal is not None
    assert tier.name == "human"


def test_router_needs_tiers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LadderRouter([], human=None, policy=Policy(PolicyConfig()), irreversible_confidence=0.9)
