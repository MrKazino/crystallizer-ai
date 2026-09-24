"""Extension kernel: typed registries and plugin loading.

A plugin is any object with a ``name`` and a ``register(context)`` method. Through the narrow
:class:`PluginContext` it can add tools, ladder tiers (external agent systems), model providers
and event handlers. Plugins come from two sources:

* explicit objects passed to :class:`~crystallizer.api.Harness` (embedding use); and
* installed packages exposing the ``crystallizer.plugins`` entry-point group, which are imported
  **only** when named in ``[plugins] enabled``. An unknown name is a configuration error.

Built-in names cannot be overridden. Plugin tools are irreversible unless trusted in config.
Generated code is never imported: plugins are packages the user installed and named.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from typing import Any, Generic, Protocol, TypeVar

from crystallizer.config import Config
from crystallizer.errors import ConfigError, ExtensionError
from crystallizer.events import EventBus, EventHandler
from crystallizer.models import ModelClient
from crystallizer.schemas import ToolSpec
from crystallizer.tiers import TierFactory
from crystallizer.tools import ToolHandler, ToolRegistry

ENTRY_POINT_GROUP = "crystallizer.plugins"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

T = TypeVar("T")
ModelFactory = Callable[[Config], ModelClient]


class Registry(Generic[T]):
    """Name-to-component map that rejects duplicates and invalid names."""

    def __init__(self, kind: str) -> None:
        """Create an empty registry for components of ``kind``."""
        self.kind = kind
        self._items: dict[str, T] = {}
        self._builtin: set[str] = set()

    def register(self, name: str, item: T, *, builtin: bool = False) -> None:
        """Add ``item`` under ``name``."""
        if not _NAME.match(name):
            raise ExtensionError(f"invalid {self.kind} name {name!r}")
        if name in self._items:
            raise ExtensionError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item
        if builtin:
            self._builtin.add(name)

    def get(self, name: str) -> T:
        """Return the component named ``name``."""
        if name not in self._items:
            known = ", ".join(sorted(self._items)) or "none"
            raise ConfigError(f"unknown {self.kind} {name!r} (known: {known})")
        return self._items[name]

    def names(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._items)

    def is_builtin(self, name: str) -> bool:
        """True if ``name`` was registered as a built-in."""
        return name in self._builtin

    def __contains__(self, name: object) -> bool:
        """Membership test by name."""
        return name in self._items


class PluginContext:
    """The only surface a plugin can touch."""

    def __init__(
        self,
        *,
        tools: ToolRegistry,
        models: Registry[ModelFactory],
        tiers: Registry[TierFactory],
        bus: EventBus,
    ) -> None:
        """Wrap the registries a plugin may extend."""
        self._tools = tools
        self._models = models
        self._tiers = tiers
        self._bus = bus

    def add_tool(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool (irreversible unless trusted in ``[policy] reversible_tools``)."""
        self._tools.register(spec, handler)

    def add_model_provider(self, name: str, factory: ModelFactory) -> None:
        """Register a model provider selectable with ``[model] provider``."""
        self._models.register(name, factory)

    def add_tier(self, name: str, factory: TierFactory) -> None:
        """Register a ladder tier usable in ``[router] ladder``."""
        self._tiers.register(name, factory)

    def subscribe(self, handler: EventHandler) -> None:
        """Observe harness events."""
        self._bus.subscribe(handler)


class Plugin(Protocol):
    """A crystallizer plugin."""

    @property
    def name(self) -> str:
        """Plugin name (what ``[plugins] enabled`` refers to)."""
        ...

    def register(self, context: PluginContext) -> None:
        """Register components through ``context``."""
        ...


def installed_plugins() -> dict[str, Callable[[], Any]]:
    """Return loaders for every installed ``crystallizer.plugins`` entry point, by name."""
    return {entry.name: entry.load for entry in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def load_plugins(
    enabled: Sequence[str],
    explicit: Iterable[Plugin],
    context: PluginContext,
    discover: Callable[[], dict[str, Callable[[], Any]]] = installed_plugins,
) -> list[str]:
    """Register explicit plugins, then the enabled entry-point plugins; return their names."""
    loaded: list[str] = []
    for plugin in explicit:
        plugin.register(context)
        loaded.append(plugin.name)
    if not enabled:
        return loaded
    available = discover()
    for name in enabled:
        if name in loaded:
            continue
        if name not in available:
            raise ConfigError(f"plugin {name!r} is enabled but not installed")
        target = available[name]()
        plugin_obj = target() if isinstance(target, type) else target
        register = getattr(plugin_obj, "register", None)
        if not callable(register):
            raise ExtensionError(f"plugin {name!r} has no register(context) method")
        register(context)
        loaded.append(name)
    return loaded
