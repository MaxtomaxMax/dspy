from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_baseline.chunking import build_corpus_from_glob
from rag_baseline.jina_embedder import JinaCodeEmbedder
from rag_baseline.types import CodeChunk


def save_chunks(chunks: list[CodeChunk], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(chunks_path: Path) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(CodeChunk.from_dict(json.loads(line)))
    return chunks


def build_and_save_index(
    *,
    repo_root: Path,
    source_glob: str,
    output_dir: Path,
    embedder: JinaCodeEmbedder,
    batch_size: int = 8,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_corpus_from_glob(repo_root, source_glob)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode_documents(texts, batch_size=batch_size)

    chunks_path = output_dir / "chunks.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"

    save_chunks(chunks, chunks_path)
    np.save(embeddings_path, embeddings)

    metadata = {
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
        "source_glob": source_glob,
        "model_name": embedder.model_name,
        "batch_size": batch_size,
        "repo_root": str(repo_root),
        "chunk_type_counts": _count_chunk_types(chunks),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _count_chunk_types(chunks: list[CodeChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.chunk_type] = counts.get(chunk.chunk_type, 0) + 1
    return counts
