"""Append-only audit logger for MCP tool calls (Task 2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _default_log_path() -> Path:
    return Path(__file__).resolve().parent.parent / "audit_log.jsonl"


def log_tool_call(
    tool_name: str,
    arguments: dict,
    result: dict,
    log_path: Path | None = None,
) -> None:
    path = log_path or _default_log_path()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
