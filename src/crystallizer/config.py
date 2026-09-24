"""TOML configuration with strict validation and named profiles.

Resolution order: built-in defaults, then the TOML file (``crystallizer.toml`` in the workspace or
``--config PATH``), then ``[profile.NAME]`` overrides when ``--profile NAME`` is given. Unknown keys
anywhere (including inside unused profiles) are errors. A missing default file means defaults; a
missing explicit ``--config`` file is an error.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from crystallizer.errors import ConfigError

DEFAULT_CONFIG_NAME = "crystallizer.toml"
BUILTIN_TIERS = ("skill", "small", "large", "human")

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "e2e": {"skills": {"min_repeats": 2, "min_shadow_runs": 5}},
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceConfig(_Section):
    """Workspace layout."""

    state_dir: str = ".crystallizer"


class ModelConfig(_Section):
    """Model provider and identifiers. Identifiers are set by the user for real providers."""

    provider: str = Field(default="mock", pattern=r"^[a-z][a-z0-9_]{0,31}$")
    small: str = "small"
    large: str = "large"
    max_tokens: int = Field(default=1024, ge=1)
    planner_tier: str = Field(default="large", pattern=r"^(small|large)$")
    mock_script: str = ""


class CostConfig(_Section):
    """Prices per million tokens. Defaults are illustrative units, not real prices."""

    small_in: float = Field(default=0.25, ge=0)
    small_out: float = Field(default=1.25, ge=0)
    large_in: float = Field(default=3.0, ge=0)
    large_out: float = Field(default=15.0, ge=0)


class ContextConfig(_Section):
    """Context budget. Estimator: ``ceil(len(text) / chars_per_token)``."""

    budget_tokens: int = Field(default=6000, ge=1)
    chars_per_token: int = Field(default=4, ge=1)
    memory_items: int = Field(default=5, ge=0)
    trace_items: int = Field(default=5, ge=0)


class RouterConfig(_Section):
    """Cost ladder and escalation thresholds."""

    samples: int = Field(default=3, ge=1)
    agreement: float = Field(default=0.67, gt=0, le=1)
    retries: int = Field(default=1, ge=0)
    irreversible_confidence: float = Field(default=0.9, ge=0, le=1)
    large_samples: int = Field(default=1, ge=1)
    ladder: list[str] = Field(default_factory=lambda: list(BUILTIN_TIERS))

    @field_validator("ladder")
    @classmethod
    def _ladder_shape(cls, ladder: list[str]) -> list[str]:
        if not ladder:
            raise ValueError("ladder must not be empty")
        if len(set(ladder)) != len(ladder):
            raise ValueError("ladder tiers must be unique")
        if "skill" in ladder and ladder[0] != "skill":
            raise ValueError("'skill' must be the first tier when present")
        if "human" in ladder and ladder[-1] != "human":
            raise ValueError("'human' must be the last tier when present")
        if ladder == ["human"]:
            raise ValueError("ladder needs at least one non-human tier")
        return ladder


class SkillsConfig(_Section):
    """Mining, shadow-testing, promotion and demotion thresholds."""

    min_repeats: int = Field(default=3, ge=2)
    min_shadow_runs: int = Field(default=20, ge=1)
    promote_pass_rate: float = Field(default=0.98, ge=0, le=1)
    demote_window: int = Field(default=20, ge=1)
    demote_pass_rate: float = Field(default=0.95, ge=0, le=1)
    auto_mine: bool = True
    auto_promote: bool = True
    min_template_length: int = Field(default=3, ge=1)


class ToolsConfig(_Section):
    """Sandbox limits for tools."""

    allowed_executables: list[str] = Field(
        default_factory=lambda: ["python", "pytest", "ruff", "mypy", "git"]
    )
    timeout_seconds: int = Field(default=60, ge=1)
    env_allowlist: list[str] = Field(default_factory=lambda: ["PATH", "HOME", "LANG", "PYTHONPATH"])
    max_file_bytes: int = Field(default=1_000_000, ge=1)
    max_output_bytes: int = Field(default=65_536, ge=256)
    protected_paths: list[str] = Field(default_factory=lambda: [".git"])

    @field_validator("allowed_executables")
    @classmethod
    def _bare_names(cls, names: list[str]) -> list[str]:
        for name in names:
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"executable {name!r} must be a bare name, not a path")
        return names


class PolicyConfig(_Section):
    """Irreversible-action policy. Empty allow-list: every irreversible action goes to a human."""

    allow_irreversible: list[str] = Field(default_factory=list)
    reversible_tools: list[str] = Field(default_factory=list)


class BudgetConfig(_Section):
    """Hard per-run spending limits; 0 disables a limit."""

    max_cost_per_run: float = Field(default=0.0, ge=0)
    max_tokens_per_run: int = Field(default=0, ge=0)
    max_model_calls_per_run: int = Field(default=0, ge=0)


class RunnerConfig(_Section):
    """Runner limits."""

    max_open_steps: int = Field(default=12, ge=1)


class MemoryConfig(_Section):
    """Memory scoring."""

    half_life_days: float = Field(default=30.0, gt=0)


class RedactionConfig(_Section):
    """Redaction tuning."""

    min_env_value_length: int = Field(default=4, ge=1)


class PluginsConfig(_Section):
    """Entry-point plugins to load. Only named plugins are ever imported."""

    enabled: list[str] = Field(default_factory=list)


class Config(_Section):
    """Complete, validated configuration."""

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    profile: dict[str, dict[str, Any]] = Field(default_factory=dict)
    active_profile: str | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` merged in recursively (neither input is modified)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate(data: dict[str, Any], origin: str) -> Config:
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"invalid configuration ({origin}): {details}") from None


def load_config(
    workspace: Path, config_path: Path | None = None, profile: str | None = None
) -> Config:
    """Load, merge and validate configuration for ``workspace``."""
    data: dict[str, Any] = {}
    origin = "defaults"
    path = config_path if config_path is not None else workspace / DEFAULT_CONFIG_NAME
    if config_path is not None and not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    if path.is_file():
        origin = str(path)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {path}: {exc}") from None
    if "active_profile" in data:
        raise ConfigError("'active_profile' is set by --profile, not in the file")
    file_profiles = data.pop("profile", {})
    if not isinstance(file_profiles, dict):
        raise ConfigError("[profile] must be a table of named profiles")
    profiles = deep_merge(BUILTIN_PROFILES, file_profiles)
    base = _validate(data, origin)
    for name, overrides in sorted(profiles.items()):
        if not isinstance(overrides, dict):
            raise ConfigError(f"profile {name!r} must be a table")
        if "profile" in overrides or "active_profile" in overrides:
            raise ConfigError(f"profile {name!r} may not define profiles")
        _validate(deep_merge(data, overrides), f"{origin} [profile.{name}]")
    if profile is None:
        return base.model_copy(update={"profile": profiles})
    if profile not in profiles:
        known = ", ".join(sorted(profiles)) or "none"
        raise ConfigError(f"unknown profile {profile!r} (known: {known})")
    merged = _validate(deep_merge(data, profiles[profile]), f"{origin} [profile.{profile}]")
    return merged.model_copy(update={"profile": profiles, "active_profile": profile})


def resolve_state_dir(workspace: Path, config: Config) -> Path:
    """Return the absolute state directory (relative paths resolve against the workspace)."""
    state = Path(config.workspace.state_dir)
    if not state.is_absolute():
        state = workspace / state
    return state.resolve()


DEFAULT_CONFIG_TEXT = """\
# crystallizer configuration. Unknown keys are errors. See docs/SPEC.md section 4.

[workspace]
state_dir = ".crystallizer"

[model]
provider = "mock"            # "mock", "anthropic", or a plugin provider name
small = "small"              # model identifiers, set by the user for real providers
large = "large"
max_tokens = 1024

[cost]                       # per million tokens, illustrative units: set real prices
small_in = 0.25
small_out = 1.25
large_in = 3.0
large_out = 15.0

[context]
budget_tokens = 6000
chars_per_token = 4          # estimator: ceil(len(text) / chars_per_token)

[router]
samples = 3
agreement = 0.67
retries = 1
irreversible_confidence = 0.9

[skills]
min_repeats = 3
min_shadow_runs = 20
promote_pass_rate = 0.98
demote_window = 20
demote_pass_rate = 0.95

[tools]
allowed_executables = ["python", "pytest", "ruff", "mypy", "git"]
timeout_seconds = 60
env_allowlist = ["PATH", "HOME", "LANG", "PYTHONPATH"]
max_file_bytes = 1000000

[policy]
allow_irreversible = []      # explicit allow-list of action names, empty by default

[budget]                     # 0 disables a limit
max_cost_per_run = 0.0
max_tokens_per_run = 0
max_model_calls_per_run = 0

[profile.e2e]
skills.min_repeats = 2
skills.min_shadow_runs = 5
"""
