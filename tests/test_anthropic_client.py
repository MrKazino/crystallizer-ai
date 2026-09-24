"""AnthropicClient without the anthropic package: fake SDK clients only, never the network."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from crystallizer.config import ModelConfig
from crystallizer.errors import ModelError
from crystallizer.models import AnthropicClient
from crystallizer.schemas import Message
from tests.helpers import open_harness, project_script


@dataclass
class Block:
    type: str
    text: str = ""


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeResponse:
    content: list[Block]
    usage: FakeUsage


@dataclass
class FakeMessages:
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail: bool = False

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider said: secret body")
        return FakeResponse(
            content=[Block("text", '{"done": '), Block("tool_use"), Block("text", "true}")],
            usage=FakeUsage(input_tokens=12, output_tokens=3),
        )


@dataclass
class FakeSDK:
    messages: FakeMessages = field(default_factory=FakeMessages)


MESSAGES = [Message(role="system", content="sys"), Message(role="user", content="hello")]


def test_complete_maps_tiers_and_usage() -> None:
    sdk = FakeSDK()
    client = AnthropicClient(
        ModelConfig(small="model-s", large="model-l"), client_factory=lambda: sdk
    )
    completion = client.complete(MESSAGES, "large", 64)
    assert completion.text == '{"done": true}'
    assert (completion.tokens_in, completion.tokens_out) == (12, 3)
    call = sdk.messages.calls[0]
    assert call["model"] == "model-l"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["max_tokens"] == 64
    client.complete(MESSAGES, "small", 8)
    assert sdk.messages.calls[1]["model"] == "model-s"


def test_provider_errors_hide_bodies() -> None:
    sdk = FakeSDK(FakeMessages(fail=True))
    client = AnthropicClient(ModelConfig(), client_factory=lambda: sdk)
    with pytest.raises(ModelError) as info:
        client.complete(MESSAGES, "small", 8)
    assert "secret body" not in info.value.message
    assert "RuntimeError" in info.value.message
    with pytest.raises(ModelError, match="unknown model tier"):
        client.complete(MESSAGES, "medium", 8)


def test_missing_package_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_package(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", no_package)
    with pytest.raises(ModelError, match="pip install"):
        AnthropicClient(ModelConfig(), environ={}).complete(MESSAGES, "small", 8)

    class Stub:
        @staticmethod
        def Anthropic(api_key: str) -> FakeSDK:  # noqa: N802 - mirrors the SDK constructor
            assert api_key == "k-from-env"
            return FakeSDK()

    monkeypatch.setattr(importlib, "import_module", lambda _name: Stub)
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(ModelConfig(), environ={}).complete(MESSAGES, "small", 8)
    client = AnthropicClient(ModelConfig(), environ={"ANTHROPIC_API_KEY": "k-from-env"})
    assert client.complete(MESSAGES, "small", 8).tokens_in == 12


def test_anthropic_provider_is_selectable(workspace: Path) -> None:
    (workspace / "crystallizer.toml").write_text(
        '[model]\nprovider = "anthropic"\n', encoding="utf-8"
    )
    from crystallizer.api import Harness
    from crystallizer.clock import FixedClock

    with Harness.open(workspace, clock=FixedClock(), environ={}) as harness:
        assert isinstance(harness.model, AnthropicClient)
    with open_harness(workspace, project_script()) as harness:
        assert not isinstance(harness.model, AnthropicClient)
