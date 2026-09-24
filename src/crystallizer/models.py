"""Model access: the :class:`ModelClient` protocol, the scripted :class:`MockModel`, and costs.

All model traffic goes through ``ModelClient.complete(messages, tier, max_tokens)``. Requests
carry one machine-readable line, ``CRYSTALLIZER-REQUEST: {json}``, which real models ignore as
context and :class:`MockModel` uses to look up its script. Tests and the benchmark only ever use
:class:`MockModel`; real providers are optional.
"""

from __future__ import annotations

import importlib
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field, model_validator

from crystallizer.clock import make_rng
from crystallizer.config import CostConfig, ModelConfig
from crystallizer.errors import ModelError
from crystallizer.schemas import Action, Completion, Message, Strict

REQUEST_MARKER = "CRYSTALLIZER-REQUEST:"
MODEL_TIERS = ("small", "large")


class ModelClient(Protocol):
    """Anything that can complete a chat for a tier (``small`` or ``large``)."""

    def complete(self, messages: Sequence[Message], tier: str, max_tokens: int) -> Completion:
        """Return one completion for ``messages`` using the model mapped to ``tier``."""
        ...


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object embedded in ``text`` (models often wrap JSON in prose)."""
    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model output")


def encode_request(payload: dict[str, Any]) -> str:
    """Render the machine-readable request line embedded in prompts."""
    return f"{REQUEST_MARKER} {json.dumps(payload, sort_keys=True)}"


def decode_request(messages: Sequence[Message]) -> dict[str, Any] | None:
    """Find the last request line in ``messages`` and parse it."""
    for message in reversed(messages):
        for line in reversed(message.content.splitlines()):
            if line.startswith(REQUEST_MARKER):
                try:
                    value = json.loads(line[len(REQUEST_MARKER) :])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


class CostTable:
    """Converts token counts into cost using per-million-token prices from config."""

    def __init__(self, config: CostConfig) -> None:
        """Store prices."""
        self._prices = {
            "small": (config.small_in, config.small_out),
            "large": (config.large_in, config.large_out),
        }

    def cost(self, tier: str, tokens_in: int, tokens_out: int) -> float:
        """Return the cost of a call on ``tier``."""
        if tier not in self._prices:
            raise ModelError(f"no price configured for tier {tier!r}")
        price_in, price_out = self._prices[tier]
        return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


class MockBehavior(StrEnum):
    """How a mock tier answers a scripted step."""

    CORRECT = "correct"
    DISAGREE = "disagree"
    WRONG = "wrong"
    ERROR = "error"
    GARBAGE = "garbage"


class MockStep(Strict):
    """Scripted answer for one step of one task."""

    action: Action | None = None
    done: bool = False
    tokens_in: int = Field(default=100, ge=0)
    tokens_out: int = Field(default=50, ge=0)
    small: MockBehavior = MockBehavior.CORRECT
    large: MockBehavior = MockBehavior.CORRECT

    @model_validator(mode="after")
    def _one_answer(self) -> MockStep:
        if (self.action is None) == (not self.done):
            raise ValueError("a mock step needs exactly one of 'action' or 'done'")
        return self


class MockScript(Strict):
    """Everything a :class:`MockModel` can answer."""

    plans: dict[str, str] = Field(default_factory=dict)
    plan_tokens_in: int = Field(default=400, ge=0)
    plan_tokens_out: int = Field(default=300, ge=0)
    actions: dict[str, list[MockStep]] = Field(default_factory=dict)
    summary_text: str = "Summary of archived memory entries."
    summary_tokens_in: int = Field(default=200, ge=0)
    summary_tokens_out: int = Field(default=60, ge=0)


class MockModel:
    """Deterministic, seedable, scripted model. Never touches the network.

    ``disagree`` makes sample ``k`` return the right action only when ``k % 3 == 0`` and a
    distinct wrong variant otherwise, so a three-sample vote cannot reach agreement. ``wrong``
    makes every sample agree on the same wrong action. ``error`` raises :class:`ModelError`.
    ``garbage`` returns text that is not JSON.
    """

    def __init__(self, script: MockScript, seed: int = 0) -> None:
        """Create a mock that answers from ``script``."""
        self.script = script
        self._rng: random.Random = make_rng(seed)
        self._samples: dict[tuple[str, int, int, str], int] = {}
        self._suffix: dict[tuple[str, int, int], int] = {}
        self.calls: list[tuple[str, str]] = []

    def complete(self, messages: Sequence[Message], tier: str, max_tokens: int) -> Completion:
        """Answer the request embedded in ``messages``."""
        if tier not in MODEL_TIERS:
            raise ModelError(f"unknown model tier {tier!r}")
        request = decode_request(messages)
        if request is None:
            raise ModelError("mock model received no request line")
        kind = str(request.get("type", ""))
        self.calls.append((kind, tier))
        if kind == "plan":
            completion = self._plan(request)
        elif kind == "summarize":
            completion = self._summarize(request)
        elif kind == "action":
            completion = self._action(request, tier)
        else:
            raise ModelError(f"mock model cannot answer request type {kind!r}")
        return completion.model_copy(update={"tokens_out": min(completion.tokens_out, max_tokens)})

    def _plan(self, request: dict[str, Any]) -> Completion:
        goal = str(request.get("goal", ""))
        if goal not in self.script.plans:
            raise ModelError("mock model has no plan scripted for this goal")
        return Completion(
            text=self.script.plans[goal],
            tokens_in=self.script.plan_tokens_in,
            tokens_out=self.script.plan_tokens_out,
        )

    def _summarize(self, request: dict[str, Any]) -> Completion:
        count = len(request.get("entries", []))
        return Completion(
            text=f"{self.script.summary_text} ({count} entries)",
            tokens_in=self.script.summary_tokens_in,
            tokens_out=self.script.summary_tokens_out,
        )

    def _action(self, request: dict[str, Any], tier: str) -> Completion:
        task_id = str(request.get("task_id", ""))
        index = int(request.get("step_index", 0))
        attempt = int(request.get("attempt", 1))
        steps = self.script.actions.get(task_id)
        if steps is None:
            raise ModelError(f"mock model has no script for task {task_id!r}")
        if index >= len(steps):
            return Completion(text=json.dumps({"done": True}), tokens_in=10, tokens_out=5)
        step = steps[index]
        behavior = step.small if tier == "small" else step.large
        sample_key = (task_id, index, attempt, tier)
        sample = self._samples.get(sample_key, 0)
        self._samples[sample_key] = sample + 1
        if behavior is MockBehavior.ERROR:
            raise ModelError(f"scripted {tier} failure for {task_id} step {index}")
        if behavior is MockBehavior.GARBAGE:
            text = "I think the next step is probably to edit a file."
        elif step.done:
            text = json.dumps({"done": True})
        elif behavior is MockBehavior.DISAGREE and sample % 3 != 0:
            text = _action_json(self._variant(step, task_id, index, sample))
        elif behavior is MockBehavior.WRONG:
            text = _action_json(self._variant(step, task_id, index, 1))
        else:
            text = _action_json(_scripted(step))
        return Completion(text=text, tokens_in=step.tokens_in, tokens_out=step.tokens_out)

    def _variant(self, step: MockStep, task_id: str, index: int, sample: int) -> Action:
        scripted = _scripted(step)
        key = (task_id, index, sample)
        if key not in self._suffix:
            self._suffix[key] = self._rng.randrange(1_000_000)
        suffix = f"-alt{sample}-{self._suffix[key]}"
        args = dict(scripted.args)
        for name in sorted(args):
            value = args[name]
            if isinstance(value, str):
                args[name] = value + suffix
                return Action(tool=scripted.tool, args=args)
            if isinstance(value, list) and value:
                args[name] = [*value[:-1], value[-1] + suffix]
                return Action(tool=scripted.tool, args=args)
        args["variant"] = suffix
        return Action(tool=scripted.tool, args=args)


def _scripted(step: MockStep) -> Action:
    if step.action is None:
        raise ModelError("scripted step has no action")
    return step.action


def _action_json(action: Action) -> str:
    return json.dumps({"tool": action.tool, "args": action.args}, sort_keys=True)


class AnthropicClient:
    """Anthropic models via the optional ``anthropic`` package.

    The package is imported lazily (never by tests or by default) and a clear error explains how
    to install it. The API key is read from ``ANTHROPIC_API_KEY`` in the environment only. Model
    identifiers come from ``[model] small`` / ``[model] large``. Request and response bodies are
    never logged; provider errors are reported by exception type only.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        client_factory: Callable[[], Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Configure the client; ``client_factory`` lets tests inject a fake SDK client."""
        self._config = config
        self._factory = client_factory
        self._environ = os.environ if environ is None else environ
        self._client: Any = None

    def _sdk_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._factory is not None:
            self._client = self._factory()
            return self._client
        try:
            sdk = importlib.import_module("anthropic")
        except ImportError:
            raise ModelError(
                "the 'anthropic' package is not installed: pip install 'crystallizer-ai[anthropic]'"
            ) from None
        api_key = self._environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ModelError("ANTHROPIC_API_KEY is not set in the environment")
        self._client = sdk.Anthropic(api_key=api_key)
        return self._client

    def complete(self, messages: Sequence[Message], tier: str, max_tokens: int) -> Completion:
        """Send one Messages API request for ``tier``."""
        model = model_identifier(self._config, tier)
        client = self._sdk_client()
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        conversation = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        try:
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=system, messages=conversation
            )
        except Exception as exc:  # noqa: BLE001 - never echo provider payloads
            raise ModelError(f"anthropic request failed ({type(exc).__name__})") from None
        text = "".join(
            str(getattr(block, "text", ""))
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = response.usage
        return Completion(
            text=text, tokens_in=int(usage.input_tokens), tokens_out=int(usage.output_tokens)
        )


def model_identifier(config: ModelConfig, tier: str) -> str:
    """Return the configured model identifier for ``tier``."""
    if tier == "small":
        return config.small
    if tier == "large":
        return config.large
    raise ModelError(f"unknown model tier {tier!r}")
