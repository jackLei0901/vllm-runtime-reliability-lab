from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "prompt",
    "prompt_token_ids",
    "input",
    "messages",
    "text",
    "json",
    "regex",
    "grammar",
    "structural_tag",
    "authorization",
    "api_key",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def anonymize_id(value: str) -> str:
    """Hash an identifier for private lab metadata, not shareable artifacts."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def redact(value: Any, key: str | None = None) -> Any:
    """Secondary canary guard for private lab output."""
    if key and key.lower() in SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return [redact(v) for v in value]
    return value


def atomic_write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Write private lab metadata; never use this for shareable incidents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(redact(payload), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)
