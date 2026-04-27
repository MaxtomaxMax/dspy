from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_baseline.config import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, DEFAULT_REACT_MAX_ITERS
from rag_baseline.jina_embedder import JinaCodeEmbedder
from rag_baseline.retriever import DenseCodeRetriever


class DenseRAGAgent:
    def __init__(
        self,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        model_name: str | None = None,
        device: str | None = None,
        max_iters: int = DEFAULT_REACT_MAX_ITERS,
    ):
        import dspy

        embedder = JinaCodeEmbedder(model_name=model_name or DEFAULT_MODEL_NAME, device=device)
        self.retriever = DenseCodeRetriever(index_dir=index_dir, embedder=embedder)
        self.react = dspy.ReAct("question -> answer", tools=[self.vector_search, self.open_chunk], max_iters=max_iters)

    def vector_search(self, query: str, top_k: int = 8) -> str:
        hits = self.retriever.search(query, top_k=top_k)
        return json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2)

    def open_chunk(self, chunk_id: str) -> str:
        chunk = self.retriever.open_chunk(chunk_id)
        return json.dumps(chunk.to_dict(), ensure_ascii=False, indent=2)

    def answer(self, question: str):
        return self.react(question=question)


def configure_dspy_lm(
    *,
    lm_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    track_usage: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    import dspy

    kwargs: dict[str, Any] = {
        "model": lm_model,
        "temperature": temperature,
        "cache": False,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm, track_usage=track_usage)
