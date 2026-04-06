"""
NOVA Product Knowledge RAG — importable module for Task 3 & Task 5.

Uses ChromaDB + dense embeddings + BM25 hybrid scoring + optional cross-encoder re-ranking.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def load_mock_catalog(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _project_root() / "nova_mock_db.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("products", [])


def products_to_chunks(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for p in products:
        sku = p.get("sku", "")
        text = (
            f"SKU {sku}. {p.get('name', '')}. Category: {p.get('category', '')}. "
            f"Ingredients: {', '.join(p.get('ingredients') or [])}. "
            f"Sizing / guide: {p.get('size_guide', '')}. "
            f"Compatibility: {p.get('compat_notes', '')}"
        )
        chunks.append({"id": sku, "text": text, "metadata": {"sku": sku, "category": p.get("category", "")}})
    return chunks


@dataclass
class HybridRAG:
    embed_model_name: str = DEFAULT_EMBED_MODEL
    rerank_model_name: str = DEFAULT_RERANK_MODEL
    persist_dir: str | None = None
    collection_name: str = "nova_products"

    def __post_init__(self) -> None:
        self.persist_dir = self.persist_dir or str(_project_root() / ".chroma" / "nova_kb")
        os.makedirs(self.persist_dir, exist_ok=True)
        self._encoder = SentenceTransformer(self.embed_model_name)
        self._reranker = CrossEncoder(self.rerank_model_name)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] | None = None
        self._chunk_ids: list[str] = []
        self._id_to_text: dict[str, str] = {}

    def index_products(self, products: list[dict[str, Any]] | None = None) -> int:
        chunks = products_to_chunks(products or load_mock_catalog())
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self._encoder.encode(texts, show_progress_bar=False).tolist()
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._tokenized_corpus = tokenized
        self._chunk_ids = ids
        self._id_to_text = dict(zip(ids, texts, strict=False))
        return len(chunks)

    def _dense_search(self, query: str, k: int = 8) -> list[tuple[str, float]]:
        qemb = self._encoder.encode([query], show_progress_bar=False).tolist()[0]
        res = self._collection.query(
            query_embeddings=[qemb],
            n_results=min(k, max(1, self._collection.count())),
            include=["documents", "distances"],
        )
        out: list[tuple[str, float]] = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for doc_id, dist in zip(res["ids"][0], res["distances"][0], strict=False):
            # cosine distance in Chroma — lower is better; convert to similarity-ish score
            sim = 1.0 - float(dist)
            out.append((doc_id, sim))
        return out

    def _sparse_search(self, query: str, k: int = 8) -> list[tuple[str, float]]:
        if not self._bm25 or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[tuple[str, float]] = []
        for idx, sc in ranked:
            if sc <= 0:
                continue
            cid = self._chunk_ids[idx]
            out.append((cid, float(sc)))
        return out

    @staticmethod
    def _merge_scores(
        dense: list[tuple[str, float]],
        sparse: list[tuple[str, float]],
        alpha: float = 0.55,
    ) -> list[tuple[str, float]]:
        def norm_sparse(vals: list[float]) -> list[float]:
            if not vals:
                return vals
            m = max(vals)
            return [v / m if m else 0.0 for v in vals]

        dmap = {i: s for i, s in dense}
        s_raw = [s for _, s in sparse]
        s_norm = norm_sparse(s_raw)
        smap = {cid: s for (cid, _), s in zip(sparse, s_norm, strict=False)}
        keys = set(dmap) | set(smap)
        merged: list[tuple[str, float]] = []
        for key in keys:
            ds = dmap.get(key, 0.0)
            ss = smap.get(key, 0.0)
            merged.append((key, alpha * ds + (1 - alpha) * ss))
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.55,
        rerank_top_n: int = 12,
    ) -> list[dict[str, Any]]:
        dense = self._dense_search(query, k=rerank_top_n)
        sparse = self._sparse_search(query, k=rerank_top_n)
        merged = self._merge_scores(dense, sparse, alpha=hybrid_alpha)[:rerank_top_n]
        if not merged:
            return []
        pairs = []
        ids_order: list[str] = []
        for cid, _ in merged:
            text = self._id_to_text.get(cid)
            if text:
                pairs.append([query, text])
                ids_order.append(cid)
        if not pairs:
            return []
        scores = self._reranker.predict(pairs)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for i in ranked_idx:
            cid = ids_order[i]
            results.append(
                {
                    "id": cid,
                    "text": self._id_to_text[cid],
                    "rerank_score": float(scores[i]),
                }
            )
        return results


def build_default_index(persist_dir: str | None = None) -> HybridRAG:
    rag = HybridRAG(persist_dir=persist_dir)
    n = rag.index_products()
    if n == 0:
        raise RuntimeError("No products indexed — check nova_mock_db.json")
    return rag
