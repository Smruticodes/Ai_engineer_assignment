#!/usr/bin/env python3
"""
Compound demo: order lookup → return ticket → escalation (Task 2).
Writes entries to ../audit_log.jsonl via the same audit helper the MCP server uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task2_mcp.audit import log_tool_call
from task2_mcp import backend_tools as bt


def run_scenario() -> list[dict]:
    steps: list[dict] = []
    r1 = bt.get_order_status("ord_10042")
    log_tool_call("get_order_status", {"order_id": "ord_10042"}, r1)
    steps.append({"step": 1, "tool": "get_order_status", "result": r1})

    r2 = bt.create_return_ticket("ord_10042", "Damaged outer box; product intact")
    log_tool_call(
        "create_return_ticket",
        {"order_id": "ord_10042", "reason": "Damaged outer box; product intact"},
        r2,
    )
    steps.append({"step": 2, "tool": "create_return_ticket", "result": r2})

    r3 = bt.get_product_by_sku("SKN-SER-01")
    log_tool_call("get_product_by_sku", {"sku": "SKN-SER-01"}, r3)
    steps.append({"step": 3, "tool": "get_product_by_sku", "result": r3})

    r4 = bt.recommend_products("sensitive", category="skincare")
    log_tool_call("recommend_products", {"skin_type": "sensitive", "category": "skincare"}, r4)
    steps.append({"step": 4, "tool": "recommend_products", "result": r4})

    r5 = bt.escalate_to_human(
        "cust_001",
        "Customer requests expedited re-shipment after carrier delay; return ticket opened.",
        priority="high",
    )
    log_tool_call(
        "escalate_to_human",
        {
            "customer_id": "cust_001",
            "summary": "Customer requests expedited re-shipment after carrier delay; return ticket opened.",
            "priority": "high",
        },
        r5,
    )
    steps.append({"step": 5, "tool": "escalate_to_human", "result": r5})
    return steps


if __name__ == "__main__":
    out = run_scenario()
    print(json.dumps({"ok": True, "scenario_steps": out}, indent=2))
