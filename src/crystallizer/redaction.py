"""Secret redaction for logs, traces, memory, checkpoints, prompts and error messages.

Redaction replaces secrets with ``[REDACTED]``. It covers:

* provider tokens: ``sk-...``, ``gh[pousr]_...``, ``AKIA...`` (with a left boundary so that, for
  example, ``task-add-module-parser`` is not mistaken for an ``sk-`` key);
* ``Bearer <token>`` credentials;
* values of sensitive keys in ``key=value``, ``key: value`` and JSON forms, where a key is
  sensitive if one of its components is or ends with ``password``, ``passwd``, ``secret``,
  ``token`` or ``apikey`` (or is ``pwd``, or is ``api`` followed by ``key``). ``access_token`` is
  sensitive; ``tokens_in`` and ``max_tokens`` are not;
* every value loaded from the workspace ``.env`` file (values shorter than the configured
  minimum length, or contained in the marker itself, are skipped because they cannot be
  meaningfully secret and would corrupt unrelated text).

All patterns are linear time: possessive quantifiers and left-boundary look-behinds prevent
backtracking, and value scans never overlap. :meth:`Redactor.redact` iterates to a fixpoint, so
it is idempotent by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}+"),
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}+"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}"),
)
_BEARER = re.compile(r"(?<![A-Za-z0-9])(?P<word>[Bb][Ee][Aa][Rr][Ee][Rr])(?P<gap>\s++)\S++")
_KEY = re.compile(
    r"""(?<![A-Za-z0-9_.\-])(?P<q>["']?+)(?P<key>[A-Za-z0-9_.\-]++)(?P=q)"""
    r"""(?P<sep>[ \t]*+[:=][ \t]*+)"""
)
_VALUE = re.compile(r"""(?P<dq>"[^"\n]*+")|(?P<sq>'[^'\n]*+')|(?P<raw>["']?+[^\s"',;&]++)""")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SPLIT = re.compile(r"[_.\-]+")
_SENSITIVE_SUFFIXES = ("password", "passwd", "secret", "secrets", "token", "apikey")
_MAX_PASSES = 8


def is_sensitive_key(key: str) -> bool:
    """Return True if ``key`` names a credential (see module docstring for the rule)."""
    parts = [part for part in _KEY_SPLIT.split(_CAMEL.sub("_", key).lower()) if part]
    for index, part in enumerate(parts):
        if part == "pwd" or part.endswith(_SENSITIVE_SUFFIXES):
            return True
        if part == "api" and index + 1 < len(parts) and parts[index + 1] == "key":
            return True
    return False


def load_env_values(path: Path) -> list[str]:
    """Return the values defined in a ``.env`` file (``KEY=VALUE`` lines, optional ``export``)."""
    if not path.is_file():
        return []
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        _, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            values.append(value)
    return values


class Redactor:
    """Applies all redaction rules. Instances are immutable once built."""

    def __init__(self, literals: Iterable[str] = (), min_literal_length: int = 4) -> None:
        """Create a redactor that also removes the given literal secret values."""
        usable = {
            value
            for value in literals
            if len(value) >= min_literal_length and value not in REDACTED and REDACTED not in value
        }
        self._literals: tuple[str, ...] = tuple(sorted(usable, key=lambda v: (-len(v), v)))

    @property
    def literal_count(self) -> int:
        """Number of literal secret values this redactor removes."""
        return len(self._literals)

    def redact(self, text: str) -> str:
        """Return ``text`` with every secret replaced by ``[REDACTED]``."""
        current = text
        for _ in range(_MAX_PASSES):
            updated = self._single_pass(current)
            if updated == current:
                return updated
            current = updated
        return current

    def redact_with_flag(self, text: str) -> tuple[str, bool]:
        """Return the redacted text and whether anything was changed."""
        result = self.redact(text)
        return result, result != text

    def redact_obj(self, value: Any) -> Any:
        """Recursively redact strings inside dicts and lists; sensitive keys lose string values."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, Mapping):
            result: dict[Any, Any] = {}
            for key, item in value.items():
                if isinstance(key, str) and is_sensitive_key(key):
                    result[key] = self._blank(item)
                else:
                    result[key] = self.redact_obj(item)
            return result
        if isinstance(value, list | tuple):
            return [self.redact_obj(item) for item in value]
        return value

    def _blank(self, value: Any) -> Any:
        if isinstance(value, str):
            return REDACTED
        if isinstance(value, list | tuple):
            return [self._blank(item) for item in value]
        if isinstance(value, Mapping):
            return self.redact_obj(value)
        return value

    def _single_pass(self, text: str) -> str:
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub(REDACTED, text)
        text = _BEARER.sub(lambda m: f"{m.group('word')}{m.group('gap')}{REDACTED}", text)
        text = _redact_key_values(text)
        for literal in self._literals:
            if literal in text:
                text = text.replace(literal, REDACTED)
        return text


def _redact_key_values(text: str) -> str:
    spans: list[tuple[int, int, str]] = []
    covered_until = 0
    for match in _KEY.finditer(text):
        if match.start() < covered_until or not is_sensitive_key(match.group("key")):
            continue
        value = _VALUE.match(text, match.end())
        if value is None:
            continue
        if value.group("dq") is not None:
            replacement = f'"{REDACTED}"'
        elif value.group("sq") is not None:
            replacement = f"'{REDACTED}'"
        else:
            replacement = REDACTED
        spans.append((value.start(), value.end(), replacement))
        covered_until = value.end()
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in spans:
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


_default = Redactor()


def default_redactor() -> Redactor:
    """Return the process-wide redactor used for errors and logging."""
    return _default


def configure_default(redactor: Redactor) -> None:
    """Replace the process-wide redactor (called once the workspace ``.env`` is known)."""
    global _default  # noqa: PLW0603 - single process-wide instance by design
    _default = redactor


def redact(text: str) -> str:
    """Redact ``text`` with the process-wide redactor."""
    return _default.redact(text)
