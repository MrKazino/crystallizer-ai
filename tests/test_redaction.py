"""Redaction fixtures for every pattern, linear-time checks, and property tests."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crystallizer.errors import CrystallizerError
from crystallizer.logging_setup import setup_logging
from crystallizer.redaction import (
    REDACTED,
    Redactor,
    configure_default,
    is_sensitive_key,
    load_env_values,
    redact,
)

SECRETS = {
    "openai": "sk-abcdefghijklmnop1234",
    "github": "ghp_abcdefghijklmnopqrstuvwxyz0123",
    "aws": "AKIAABCDEFGHIJKLMNOP",
}


@pytest.mark.parametrize("secret", list(SECRETS.values()))
def test_token_patterns_are_redacted(secret: str) -> None:
    out = Redactor().redact(f"value {secret} end")
    assert secret not in out
    assert out == f"value {REDACTED} end"


@pytest.mark.parametrize("prefix", ["gho_", "ghu_", "ghs_", "ghr_"])
def test_all_github_prefixes(prefix: str) -> None:
    token = prefix + "A" * 24
    assert Redactor().redact(token) == REDACTED


def test_left_boundary_prevents_false_positives() -> None:
    text = "task-add-module-parser-extra risk-assessment-framework-v2"
    assert Redactor().redact(text) == text


def test_bearer_is_redacted_case_insensitive() -> None:
    assert Redactor().redact("Authorization: Bearer abc.def") == f"Authorization: Bearer {REDACTED}"
    assert Redactor().redact("bearer xyz") == f"bearer {REDACTED}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("password=hunter2 x", f"password={REDACTED} x"),
        ("password: hunter2", f"password: {REDACTED}"),
        ("SECRET = abc", f"SECRET = {REDACTED}"),
        ("access_token=xyz", f"access_token={REDACTED}"),
        ("api-key: k1", f"api-key: {REDACTED}"),
        ("apiKey=k2", f"apiKey={REDACTED}"),
        ('{"api_key": "abc"}', f'{{"api_key": "{REDACTED}"}}'),
        ("token='q w'", f"token='{REDACTED}'"),
        ('export GITHUB_TOKEN="x1"', f'export GITHUB_TOKEN="{REDACTED}"'),
        ("x=password=secret", f"x=password={REDACTED}"),
        ('password="unterminated', f"password={REDACTED}"),
    ],
)
def test_key_value_forms(text: str, expected: str) -> None:
    assert Redactor().redact(text) == expected


@pytest.mark.parametrize(
    "text", ["tokens_in=500", "max_tokens: 1024", "tokenizer=x", "https://example.com"]
)
def test_non_sensitive_keys_untouched(text: str) -> None:
    assert Redactor().redact(text) == text


@pytest.mark.parametrize(
    ("key", "sensitive"),
    [
        ("password", True),
        ("db_passwd", True),
        ("pwd", True),
        ("client_secret", True),
        ("GITHUB_TOKEN", True),
        ("x-api-key", True),
        ("accessToken", True),
        ("tokens_in", False),
        ("max_tokens", False),
        ("keyboard", False),
    ],
)
def test_is_sensitive_key(key: str, sensitive: bool) -> None:
    assert is_sensitive_key(key) is sensitive


def test_env_file_values_are_redacted(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nDB_URL=postgres://u:pw@h/db\nexport NAME='quoted value'\nSHORT=ab\nEMPTY=\n"
        "not a pair\n",
        encoding="utf-8",
    )
    values = load_env_values(env)
    assert values == ["postgres://u:pw@h/db", "quoted value", "ab"]
    redactor = Redactor(values, min_literal_length=4)
    assert redactor.literal_count == 2
    out = redactor.redact("url postgres://u:pw@h/db and quoted value and ab")
    assert out == f"url {REDACTED} and {REDACTED} and ab"


def test_missing_env_file(tmp_path: Path) -> None:
    assert load_env_values(tmp_path / ".env") == []


def test_literals_inside_marker_are_skipped() -> None:
    redactor = Redactor(["REDACTED", "EDAC", f"x{REDACTED}"])
    assert redactor.literal_count == 0


def test_redact_obj_blanks_sensitive_keys() -> None:
    data = {
        "password": "p",
        "tokens": ["a", "b"],
        "api_key": ["x"],
        "nested": {"secret": {"inner": "sk-abcdefghijklmnop1234"}, "n": 3},
        "count": 1,
    }
    out = Redactor().redact_obj(data)
    assert out["password"] == REDACTED
    assert out["tokens"] == ["a", "b"]
    assert out["api_key"] == [REDACTED]
    assert out["nested"]["secret"]["inner"] == REDACTED
    assert out["count"] == 1
    assert Redactor().redact_obj(("sk-abcdefghijklmnop1234",)) == [REDACTED]
    assert Redactor().redact_obj({"secret": 5}) == {"secret": 5}


def test_redact_with_flag() -> None:
    assert Redactor().redact_with_flag("password=x") == (f"password={REDACTED}", True)
    assert Redactor().redact_with_flag("plain") == ("plain", False)


@pytest.mark.parametrize(
    "adversarial",
    [
        "a" * 200_000,
        "a-" * 100_000,
        'password="' * 20_000,
        'a="' * 50_000,
        "password=password=" * 20_000,
        "Bearer " * 50_000,
        "sk-" * 60_000,
        '"a"' * 50_000,
        "x" * 100_000 + "password=" + "y" * 100_000,
    ],
)
def test_long_inputs_complete(adversarial: str) -> None:
    out = Redactor().redact(adversarial)
    assert isinstance(out, str)


def test_error_messages_are_redacted() -> None:
    error = CrystallizerError("failed with token=abc123 and sk-abcdefghijklmnop1234")
    assert "abc123" not in error.message
    assert "sk-abc" not in str(error)


def test_default_redactor_is_configurable() -> None:
    configure_default(Redactor(["hunter22"]))
    assert redact("pw hunter22") == f"pw {REDACTED}"


def test_logs_redact_messages_and_extra_fields() -> None:
    stream = io.StringIO()
    logger = setup_logging(verbose=True, stream=stream)
    logger.info(
        "calling with Bearer abc", extra={"payload": {"api_key": "zzz"}, "note": "password=p"}
    )
    line = json.loads(stream.getvalue())
    assert line["message"] == f"calling with Bearer {REDACTED}"
    assert line["payload"] == {"api_key": REDACTED}
    assert line["note"] == f"password={REDACTED}"
    logging.getLogger("crystallizer").handlers.clear()


_text = st.text(
    alphabet=st.sampled_from(list("abcAB01_-=: \"'\n.sk-ghpAKIBearer[]{}")), max_size=120
)


@given(_text)
def test_redaction_never_raises_and_is_idempotent(text: str) -> None:
    redactor = Redactor(["secretvalue", "abc=def"])
    once = redactor.redact(text)
    assert redactor.redact(once) == once


@given(st.text(max_size=200))
def test_redaction_is_idempotent_on_any_unicode(text: str) -> None:
    once = Redactor().redact(text)
    assert Redactor().redact(once) == once
