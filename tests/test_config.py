"""Configuration loading, strict keys, and profiles."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from crystallizer.config import (
    DEFAULT_CONFIG_TEXT,
    Config,
    deep_merge,
    load_config,
    resolve_state_dir,
)
from crystallizer.errors import ConfigError, ExitCode


def write(workspace: Path, text: str) -> Path:
    path = workspace / "crystallizer.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_means_defaults(workspace: Path) -> None:
    config = load_config(workspace)
    assert config.router.samples == 3
    assert config.skills.min_repeats == 3
    assert config.cost.large_out == 15.0
    assert config.context.chars_per_token == 4
    assert config.policy.allow_irreversible == []
    assert "e2e" in config.profile
    assert config.active_profile is None


def test_default_config_text_matches_defaults(workspace: Path) -> None:
    write(workspace, DEFAULT_CONFIG_TEXT)
    loaded = load_config(workspace)
    defaults = Config()
    for section in ("workspace", "model", "cost", "context", "router", "skills", "tools"):
        assert getattr(loaded, section) == getattr(defaults, section)


def test_builtin_e2e_profile_lowers_thresholds_only_there(workspace: Path) -> None:
    config = load_config(workspace, profile="e2e")
    assert config.skills.min_repeats == 2
    assert config.skills.min_shadow_runs == 5
    assert config.active_profile == "e2e"
    assert load_config(workspace).skills.min_repeats == 3


def test_file_profile_overrides(workspace: Path) -> None:
    write(workspace, "[router]\nsamples = 5\n[profile.fast]\nrouter.samples = 1\n")
    assert load_config(workspace).router.samples == 5
    assert load_config(workspace, profile="fast").router.samples == 1


@pytest.mark.parametrize(
    "text",
    [
        "[router]\nsamplez = 3\n",
        "[unknown]\nx = 1\n",
        "[router]\nagreement = 1.5\n",
        "[profile.bad]\nrouter.nope = 1\n",
        "[profile.bad]\nprofile.x = 1\n",
        "profile = 3\n",
        "[profile]\nbad = 3\n",
        "active_profile = 'x'\n",
        '[tools]\nallowed_executables = ["/bin/rm"]\n',
        '[router]\nladder = ["small", "skill"]\n',
        '[router]\nladder = ["human", "small"]\n',
        '[router]\nladder = ["small", "small"]\n',
        "[router]\nladder = []\n",
        '[router]\nladder = ["human"]\n',
        "[skills]\nmin_repeats = 1\n",
    ],
)
def test_invalid_configs_are_usage_errors(workspace: Path, text: str) -> None:
    write(workspace, text)
    with pytest.raises(ConfigError) as info:
        load_config(workspace)
    assert info.value.exit_code is ExitCode.USAGE


def test_invalid_toml(workspace: Path) -> None:
    write(workspace, "[router\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(workspace)


def test_unknown_profile(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="unknown profile"):
        load_config(workspace, profile="nope")


def test_explicit_missing_config(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(workspace, config_path=workspace / "missing.toml")


def test_explicit_config_path(workspace: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.toml"
    other.write_text("[context]\nbudget_tokens = 99\n", encoding="utf-8")
    assert load_config(workspace, config_path=other).context.budget_tokens == 99


def test_resolve_state_dir(workspace: Path, tmp_path: Path) -> None:
    assert resolve_state_dir(workspace, Config()) == (workspace / ".crystallizer").resolve()
    absolute = Config.model_validate({"workspace": {"state_dir": str(tmp_path / "st")}})
    assert resolve_state_dir(workspace, absolute) == (tmp_path / "st").resolve()


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 1}
    merged = deep_merge(base, {"a": {"b": 5}, "e": [1]})
    assert merged == {"a": {"b": 5, "c": 2}, "d": 1, "e": [1]}
    assert base == {"a": {"b": 1, "c": 2}, "d": 1}


def test_default_text_is_valid_toml() -> None:
    assert "profile" in tomllib.loads(DEFAULT_CONFIG_TEXT)
