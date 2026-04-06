# NOVA AI Platform — AI Engineer Assessment

Multi-agent customer support and personalization stack for the fictional D2C brand **NOVA**. This repository follows the assessment layout: prompt engineering, MCP tools, hybrid RAG, QLoRA fine-tuning (Colab), and a **LangGraph** orchestrator with audit trails.

**Primary LLM:** [Groq](https://console.groq.com/) (OpenAI-compatible API). Set `GROQ_API_KEY` in your environment (see `.env.example`). The code also accepts `groq_api_key` for convenience.

## Repository layout

| Path | Purpose |
|------|---------|
| `prompts/` | Versioned system prompts (COSTAR, CoT intent) |
| `nova_mock_db.json` | Synthetic customers, orders, products |
| `task1_prompt_engineering.ipynb` | COSTAR, CoT classification, injection demo |
| `task2_mcp/` | MCP server (5 tools), stdio client, compound `demo.py` |
| `rag_module.py` | Chroma + hybrid search + cross-encoder re-rank (Task 3 & 5) |
| `task3_rag_pipeline.ipynb` | Indexing, retrieval, optional RAGAS |
| `task4_finetune.ipynb` | QLoRA / W&B / HF Hub placeholders for Colab |
| `task5_nova_platform.py` | LangGraph + Groq + tools + RAG + escalation |
| `task5_demo.py` | Sample tickets → `nova_traces.json` |
| `scripts/export_langgraph_png.py` | Renders `nova_agent_graph.png` |
| `audit_log.jsonl` | MCP tool audit log (from `task2_mcp/demo.py`) |
| `evaluation_report.json` | RAGAS-style metrics (refresh from Task 3) |
| `nova_traces.json` | Agent audit trails (from Task 5 demo) |

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY=... to .env
```

Run Task 2 demo (logs to `audit_log.jsonl`):

```bash
python task2_mcp/demo.py
```

Run Task 5 (single ticket):

```bash
python task5_nova_platform.py
```

Regenerate agent traces and LangGraph image:

```bash
python task5_demo.py
python scripts/export_langgraph_png.py
```

## Colab notebooks

Upload this repo (or clone from GitHub) and open:

- `task1_prompt_engineering.ipynb`
- `task3_rag_pipeline.ipynb`
- `task4_finetune.ipynb`

**Shareable links (fill in after you publish):**

| Asset | Link |
|-------|------|
| Task 1 notebook | Run in local |
| Task 3 notebook | Run in local |
| Task 4 Colab | *https://colab.research.google.com/drive/1zW-8zXg7TASrpcQMLBDm749gqE_4Pkod?usp=sharing* |
| model link (Task 4) | *https://drive.google.com/drive/folders/13WodG7XvZsmoWzlFRTHoMnYdF19U6frH?usp=sharing* |
| GitHub repository | *https://github.com/Smruticodes/Ai_engineer_assignment* |

## Groq configuration

- Default chat model: `llama-3.3-70b-versatile` (override with `GROQ_MODEL`).
- Python uses `https://api.groq.com/openai/v1` as the OpenAI-compatible base URL (`nova_llm.py` and notebooks).

## Security

- Never commit real API keys. Use `.env` locally and Colab secrets in the cloud.
- If a key was ever pasted into a chat or committed, rotate it in the Groq console.

## License

Assessment submission — fictional brand and data for evaluation purposes.
