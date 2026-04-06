#!/usr/bin/env python3
"""
MCP stdio client for NOVA backend tools.

Usage (from repo root, with MCP server available on PATH):
  python -m task2_mcp.client get_order_status '{"order_id": "ord_10042"}'

For Colab / quick tests, use `backend_tools` directly or run `demo.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import McpError


async def _call_tool(tool_name: str, arguments: dict) -> str:
    root = Path(__file__).resolve().parent.parent
    server_py = Path(__file__).resolve().parent / "server.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    sp = StdioServerParameters(command=sys.executable, args=[str(server_py)], env=env)
    async with stdio_client(sp) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                result = await session.call_tool(tool_name, arguments)
            except McpError as e:
                return json.dumps({"error": str(e)})
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(texts) if texts else str(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA MCP stdio client")
    parser.add_argument("tool", help="Tool name")
    parser.add_argument("args_json", help="JSON object of arguments")
    args = parser.parse_args()
    try:
        payload = json.loads(args.args_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid_json: {e}"}))
        sys.exit(1)
    out = asyncio.run(_call_tool(args.tool, payload))
    print(out)


if __name__ == "__main__":
    main()
