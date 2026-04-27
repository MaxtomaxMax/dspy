from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_baseline.config import DEFAULT_PREVIEW_CHARS, DEFAULT_TOP_K
from rag_baseline.indexer import load_chunks
from rag_baseline.jina_embedder import JinaCodeEmbedder
from rag_baseline.types import CodeChunk, RetrievalHit


class DenseCodeRetriever:
    def __init__(
        self,
        *,
        index_dir: Path,
        embedder: JinaCodeEmbedder,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
    ):
        self.index_dir = index_dir
        self.embedder = embedder
        self.preview_chars = preview_chars

        self.chunks = self._load_chunks(index_dir / "chunks.jsonl")
        self.embeddings = np.load(index_dir / "embeddings.npy")
        self.metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))

    def _load_chunks(self, chunks_path: Path) -> list[CodeChunk]:
        return load_chunks(chunks_path)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalHit]:
        query_embedding = self.embedder.encode_queries([query])[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(-scores)[:top_k]

        hits: list[RetrievalHit] = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=float(scores[idx]),
                    file_path=chunk.file_path,
                    symbol_path=chunk.symbol_path,
                    chunk_type=chunk.chunk_type,
                    preview=chunk.text[: self.preview_chars],
                )
            )
        return hits

    def open_chunk(self, chunk_id: str) -> CodeChunk:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"Unknown chunk_id: {chunk_id}")

    def format_hits(self, hits: list[RetrievalHit]) -> str:
        formatted = []
        for idx, hit in enumerate(hits, start=1):
            formatted.append(
                "\n".join(
                    [
                        f"[{idx}] score={hit.score:.4f}",
                        f"chunk_id={hit.chunk_id}",
                        f"file_path={hit.file_path}",
                        f"symbol_path={hit.symbol_path or '<module>'}",
                        f"chunk_type={hit.chunk_type}",
                        "preview:",
                        hit.preview,
                    ]
                )
            )
        return "\n\n".join(formatted)
