#!/usr/bin/env python3
"""MCP server exposing five NOVA backend tools with audit logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as script: python task2_mcp/server.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

from task2_mcp.audit import log_tool_call
from task2_mcp import backend_tools as bt

mcp = FastMCP("NOVA Backend")


def _log(tool: str, args: dict, result: dict) -> str:
    log_tool_call(tool, args, result)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Look up order status, carrier, and tracking by order_id."""
    r = bt.get_order_status(order_id)
    return _log("get_order_status", {"order_id": order_id}, r)


@mcp.tool()
def create_return_ticket(order_id: str, reason: str) -> str:
    """Create a return request for an order; returns a return_ticket_id."""
    r = bt.create_return_ticket(order_id, reason)
    return _log("create_return_ticket", {"order_id": order_id, "reason": reason}, r)


@mcp.tool()
def get_product_by_sku(sku: str) -> str:
    """Fetch product facts: ingredients, sizing, compatibility by SKU."""
    r = bt.get_product_by_sku(sku)
    return _log("get_product_by_sku", {"sku": sku}, r)


@mcp.tool()
def recommend_products(skin_type: str, category: str | None = None) -> str:
    """Return personalized product recommendations for a skin type and optional category."""
    r = bt.recommend_products(skin_type, category)
    return _log(
        "recommend_products",
        {"skin_type": skin_type, "category": category},
        r,
    )


@mcp.tool()
def escalate_to_human(customer_id: str, summary: str, priority: str = "normal") -> str:
    """Queue escalation to a human agent with context and priority."""
    r = bt.escalate_to_human(customer_id, summary, priority)
    return _log(
        "escalate_to_human",
        {"customer_id": customer_id, "summary": summary, "priority": priority},
        r,
    )


if __name__ == "__main__":
    mcp.run()
