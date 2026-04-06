"""
NOVA multi-agent platform (Task 5) — LangGraph + Groq + RAG + backend tools + audit traces.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nova_llm import groq_llm
from rag_module import HybridRAG
from task2_mcp import backend_tools as bt


def load_prompt(name: str) -> str:
    p = _ROOT / "prompts" / name
    return p.read_text(encoding="utf-8")


class NovaState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: str
    ticket_text: str
    intent: str
    escalate: bool
    rag_chunks: list[dict[str, Any]]
    final_reply: str
    audit_trail: Annotated[list[dict[str, Any]], add]


_INTENT_ORDER = re.compile(r"\b(order|track|shipping|delivered|package)\b", re.I)
_INTENT_RETURN = re.compile(r"\b(return|refund|exchange)\b", re.I)
_INTENT_PRODUCT = re.compile(r"\b(ingredient|niacinamide|size|fit|serum|lip|denim)\b", re.I)


def heuristic_precheck(text: str) -> str | None:
    if "ignore previous" in text.lower() or "system prompt" in text.lower():
        return "escalation"
    if _INTENT_ORDER.search(text):
        return "order_status"
    if _INTENT_RETURN.search(text):
        return "returns"
    if _INTENT_PRODUCT.search(text):
        return "product_knowledge"
    return None


def classify_node(state: NovaState) -> NovaState:
    llm = groq_llm()
    ticket = state.get("ticket_text") or ""
    pre = heuristic_precheck(ticket)
    if pre:
        return {
            "intent": pre,
            "escalate": pre == "escalation",
            "audit_trail": [{"step": "classify", "detail": {"method": "heuristic", "intent": pre}}],
        }

    cot = load_prompt("v1_intent_cot_instructions.txt")
    sys = SystemMessage(
        content=cot
        + "\nClassify this customer message and respond with JSON only.\nMessage:\n"
        + ticket
    )
    out = llm.invoke([sys])
    text = out.content if isinstance(out.content, str) else str(out.content)
    intent = "other"
    esc = False
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            data = json.loads(m.group())
            intent = data.get("intent", intent)
            esc = bool(data.get("escalate"))
    except json.JSONDecodeError:
        intent = "other"
    return {
        "intent": intent,
        "escalate": esc,
        "audit_trail": [
            {"step": "classify", "detail": {"method": "llm", "raw": text[:500], "intent": intent}}
        ],
    }


def route_node(state: NovaState) -> Literal["tools", "rag", "escalate", "respond"]:
    if state.get("escalate") or state.get("intent") == "escalation":
        return "escalate"
    it = state.get("intent") or ""
    if it in ("order_status", "returns"):
        return "tools"
    if it == "product_knowledge":
        return "rag"
    return "respond"


def _normalize_order_id(fragment: str) -> str:
    m = re.match(r"ord_?(\d+)", fragment.strip(), re.I)
    if m:
        return f"ord_{m.group(1)}"
    return fragment


def tools_node(state: NovaState) -> NovaState:
    """Use mock DB tools — demo uses order_id from ticket or default."""
    it = state.get("intent")
    ticket = state.get("ticket_text", "")
    oid_m = re.search(r"ord[_-]?\d+", ticket, re.I)
    order_id = _normalize_order_id(oid_m.group(0)) if oid_m else "ord_10042"
    results: dict[str, Any] = {}
    if it == "order_status":
        results = bt.get_order_status(order_id)
    elif it == "returns":
        results = bt.create_return_ticket(order_id, "Customer requested return via AI flow")
    summary = json.dumps(results, ensure_ascii=False)[:2000]
    return {
        "messages": [
            AIMessage(
                content=f"[Tool results for {it}]\n{summary}",
                name="nova_tools",
            )
        ],
        "audit_trail": [{"step": "mcp_tools", "detail": {"intent": it, "order_id": order_id, "result": results}}],
    }


def rag_node(state: NovaState) -> NovaState:
    q = state.get("ticket_text", "")
    persist = str(_ROOT / ".chroma" / "nova_kb")
    rag = HybridRAG(persist_dir=persist)
    try:
        rag.index_products()
    except Exception:
        pass
    chunks = rag.retrieve(q, top_k=3)
    ctx = "\n".join(c["text"] for c in chunks) if chunks else ""
    return {
        "rag_chunks": chunks,
        "messages": [AIMessage(content=f"[Product KB]\n{ctx}", name="nova_rag")],
        "audit_trail": [{"step": "rag_retrieval", "detail": {"query": q[:200], "chunks": chunks}}],
    }


def synthesize_node(state: NovaState) -> NovaState:
    llm = groq_llm(temperature=0.35)
    costar = load_prompt("v1_support_brain_costar.txt")
    msgs = [
        SystemMessage(content=costar),
        HumanMessage(
            content="Compose the customer-facing reply for NOVA.\nTicket:\n"
            + state.get("ticket_text", "")
        ),
    ]
    msgs.extend(state.get("messages") or [])
    out = llm.invoke(msgs)
    text = out.content if isinstance(out.content, str) else str(out.content)
    if not (text or "").strip():
        text = (
            "Thanks for your message. Here is what I found in our product knowledge base:\n\n"
            + "\n\n".join(
                (c.get("text") or "")[:500] for c in (state.get("rag_chunks") or [])[:2]
            )
        )
    return {
        "final_reply": text,
        "messages": [AIMessage(content=text, name="nova_brain")],
        "audit_trail": [{"step": "synthesize", "detail": {"model": "groq", "chars": len(text)}}],
    }


def human_handoff_node(state: NovaState) -> NovaState:
    cid = state.get("customer_id") or "cust_001"
    summ = state.get("ticket_text", "")[:500]
    r = bt.escalate_to_human(cid, summ, priority="normal")
    msg = (
        "I've connected you with a specialist who will review this shortly. "
        f"Reference: {r.get('escalation_id', 'n/a')}."
    )
    return {
        "final_reply": msg,
        "messages": [AIMessage(content=msg, name="nova_escalation")],
        "audit_trail": [{"step": "escalation", "detail": {"result": r}}],
    }


def build_graph():
    g = StateGraph(NovaState)
    g.add_node("classify", classify_node)
    g.add_node("tools", tools_node)
    g.add_node("rag", rag_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("human_handoff", human_handoff_node)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        route_node,
        {
            "tools": "tools",
            "rag": "rag",
            "escalate": "human_handoff",
            "respond": "synthesize",
        },
    )
    g.add_edge("tools", "synthesize")
    g.add_edge("rag", "synthesize")
    g.add_edge("synthesize", END)
    g.add_edge("human_handoff", END)
    return g.compile()


def run_ticket(
    ticket_text: str,
    customer_id: str = "cust_001",
) -> dict[str, Any]:
    app = build_graph()
    init: NovaState = {
        "messages": [],
        "ticket_text": ticket_text,
        "customer_id": customer_id,
        "audit_trail": [],
    }
    out = app.invoke(init)
    return {
        "final_reply": out.get("final_reply"),
        "intent": out.get("intent"),
        "audit_trail": out.get("audit_trail", []),
    }


def export_traces(path: str | Path | None = None) -> None:
    """Append last run — demo script writes merged file."""
    path = path or (_ROOT / "nova_traces.json")
    # placeholder: task5_demo overwrites with structured run
    path.write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    sample = os.getenv(
        "NOVA_SAMPLE_TICKET",
        "Where is order ord_10042? I'm worried it is lost.",
    )
    result = run_ticket(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
