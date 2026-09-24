"""Event bus isolation, registries and plugin loading (explicit and entry-point)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from crystallizer.budget import BudgetGovernor
from crystallizer.clock import FixedClock
from crystallizer.config import BudgetConfig, Config, CostConfig, PolicyConfig
from crystallizer.errors import ConfigError, ExtensionError
from crystallizer.events import EventBus, EventKind
from crystallizer.extensions import (
    ModelFactory,
    PluginContext,
    Registry,
    installed_plugins,
    load_plugins,
)
from crystallizer.models import CostTable, MockModel, MockScript, ModelClient
from crystallizer.policy import Policy
from crystallizer.redaction import REDACTED, Redactor
from crystallizer.schemas import ArgValue, Event, StepRequest, TierResult, ToolResult, ToolSpec
from crystallizer.tiers import Proposer, TierContext, TierFactory
from crystallizer.tools import Sandbox, ToolRegistry


def make_bus() -> EventBus:
    return EventBus(FixedClock(), Redactor())


def test_bus_redacts_and_isolates_failures() -> None:
    bus = make_bus()
    seen: list[Event] = []

    def boom(event: Event) -> None:
        raise RuntimeError("observer failure")

    bus.subscribe(boom)
    unsubscribe = bus.subscribe(seen.append)
    event = bus.publish(EventKind.STEP_EXECUTED, run_id="r", task_id="t", note="token=abc", n=1)
    assert bus.errors == 1
    assert seen == [event]
    assert event.data == {"note": f"token={REDACTED}", "n": 1}
    unsubscribe()
    unsubscribe()
    bus.publish(EventKind.RUN_FINISHED)
    assert len(seen) == 1


def test_registry_rules() -> None:
    registry: Registry[int] = Registry("thing")
    registry.register("one", 1, builtin=True)
    registry.register("two", 2)
    assert registry.get("one") == 1
    assert registry.names() == ["one", "two"]
    assert registry.is_builtin("one")
    assert not registry.is_builtin("two")
    assert "two" in registry
    with pytest.raises(ExtensionError):
        registry.register("one", 3)
    with pytest.raises(ExtensionError):
        registry.register("Bad Name", 3)
    with pytest.raises(ConfigError):
        registry.get("three")


class EchoTier:
    """A plugin tier standing in for an external agent system."""

    def __init__(self, context: TierContext) -> None:
        self.context = context

    @property
    def name(self) -> str:
        return "echo"

    def propose(self, request: StepRequest) -> TierResult:
        raise NotImplementedError


class DemoPlugin:
    name = "demo"

    def __init__(self) -> None:
        self.registered = False

    def register(self, context: PluginContext) -> None:
        def handler(args: Mapping[str, ArgValue], sandbox: Sandbox) -> ToolResult:
            return ToolResult(ok=True, output="demo")

        def factory(config: Config) -> ModelClient:
            return MockModel(MockScript())

        context.add_tool(
            ToolSpec(name="demo_tool", description="d", args_schema={}, reversible=True), handler
        )
        context.add_model_provider("demo_models", factory)
        context.add_tier("echo", EchoTier)
        context.subscribe(lambda _event: None)
        self.registered = True


def make_context(
    sandbox: Sandbox,
) -> tuple[PluginContext, ToolRegistry, Registry[ModelFactory], Registry[TierFactory]]:
    tools = ToolRegistry(sandbox)
    models: Registry[ModelFactory] = Registry("model provider")
    tiers: Registry[TierFactory] = Registry("tier")
    return (
        PluginContext(tools=tools, models=models, tiers=tiers, bus=make_bus()),
        tools,
        models,
        tiers,
    )


def test_explicit_plugin_registers_everything(sandbox: Sandbox) -> None:
    context, tools, models, tiers = make_context(sandbox)
    plugin = DemoPlugin()
    assert load_plugins([], [plugin], context) == ["demo"]
    assert plugin.registered
    assert tools.spec("demo_tool") is not None
    assert isinstance(models.get("demo_models")(Config()), MockModel)
    tier_context = TierContext(
        config=Config(),
        model=MockModel(MockScript()),
        costs=CostTable(CostConfig()),
        budget=BudgetGovernor(BudgetConfig()),
        policy=Policy(PolicyConfig()),
        bus=make_bus(),
        tools=tuple(tools.specs()),
    )
    tier: Proposer = tiers.get("echo")(tier_context)
    assert tier.name == "echo"


def test_entry_point_plugins_only_when_enabled(sandbox: Sandbox) -> None:
    loaded: list[str] = []

    def discover() -> dict[str, Callable[[], Any]]:
        loaded.append("discovered")
        return {"demo": lambda: DemoPlugin, "broken": object}

    context, *_ = make_context(sandbox)
    assert load_plugins([], [], context, discover=discover) == []
    assert loaded == []
    assert load_plugins(["demo"], [], context, discover=discover) == ["demo"]
    context2, *_ = make_context(sandbox)
    with pytest.raises(ConfigError, match="not installed"):
        load_plugins(["ghost"], [], context2, discover=discover)
    with pytest.raises(ExtensionError, match="register"):
        load_plugins(["broken"], [], context2, discover=discover)


def test_enabled_plugin_already_given_explicitly_is_not_loaded_twice(sandbox: Sandbox) -> None:
    context, *_ = make_context(sandbox)
    names = load_plugins(["demo"], [DemoPlugin()], context, discover=lambda: {})
    assert names == ["demo"]


def test_installed_plugins_is_a_mapping() -> None:
    assert isinstance(installed_plugins(), dict)
