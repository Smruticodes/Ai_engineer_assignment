"""Shared Groq LLM setup (OpenAI-compatible API)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def groq_llm(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY (or groq_api_key) in your environment.")
    mid = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatOpenAI(
        model=mid,
        temperature=temperature,
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )
