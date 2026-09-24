"""Canonical JSON and SHA-256 helpers shared by the journal, checkpoints and skills."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Convert pydantic models (recursively) into plain JSON-compatible data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    """Return the hex SHA-256 digest of ``data`` (strings are UTF-8 encoded)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def digest(value: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of ``value``."""
    return sha256_hex(canonical_json(value))
