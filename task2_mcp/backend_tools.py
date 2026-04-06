"""Five NOVA backend tools backed by nova_mock_db.json (no network)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_db() -> dict[str, Any]:
    path = _root() / "nova_mock_db.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_order_status(order_id: str) -> dict[str, Any]:
    db = _load_db()
    for o in db.get("orders", []):
        if o.get("order_id") == order_id:
            return {"ok": True, "order": o}
    return {"ok": False, "error": "order_not_found"}


def create_return_ticket(order_id: str, reason: str) -> dict[str, Any]:
    status = get_order_status(order_id)
    if not status.get("ok"):
        return status
    ticket_id = f"ret_{uuid.uuid4().hex[:10]}"
    return {
        "ok": True,
        "return_ticket_id": ticket_id,
        "order_id": order_id,
        "reason": reason,
        "status": "pending_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def get_product_by_sku(sku: str) -> dict[str, Any]:
    db = _load_db()
    for p in db.get("products", []):
        if p.get("sku") == sku:
            return {"ok": True, "product": p}
    return {"ok": False, "error": "sku_not_found"}


def recommend_products(skin_type: str, category: str | None = None) -> dict[str, Any]:
    """Rule-based + catalog filter — personalization stub for demo."""
    db = _load_db()
    out: list[dict[str, Any]] = []
    for p in db.get("products", []):
        if category and p.get("category") != category:
            continue
        score = 1.0
        if p.get("category") == "skincare" and skin_type == "sensitive":
            if "fragrance-free" in " ".join(p.get("ingredients") or []).lower():
                score += 0.5
        out.append({"sku": p["sku"], "name": p["name"], "score": score})
    out.sort(key=lambda x: x["score"], reverse=True)
    return {"ok": True, "recommendations": out[:5], "skin_type": skin_type}


def escalate_to_human(
    customer_id: str,
    summary: str,
    priority: str = "normal",
) -> dict[str, Any]:
    esc_id = f"esc_{uuid.uuid4().hex[:10]}"
    return {
        "ok": True,
        "escalation_id": esc_id,
        "customer_id": customer_id,
        "summary": summary,
        "priority": priority,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
