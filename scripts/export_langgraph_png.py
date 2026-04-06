#!/usr/bin/env python3
"""Write nova_agent_graph.png from the compiled LangGraph (requires graphviz)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task5_nova_platform import build_graph


def main() -> None:
    app = build_graph()
    try:
        png = app.get_graph().draw_mermaid_png()
    except Exception as e:
        print(
            "Could not render PNG (install graphviz system package or use Colab). "
            f"Error: {e}",
            file=sys.stderr,
        )
        # Fallback: write mermaid text for manual rendering
        mermaid = app.get_graph().draw_mermaid()
        Path(_ROOT / "nova_agent_graph.mmd").write_text(mermaid, encoding="utf-8")
        print(f"Wrote {_ROOT / 'nova_agent_graph.mmd'} instead.")
        sys.exit(1)
    out = _ROOT / "nova_agent_graph.png"
    out.write_bytes(png)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
