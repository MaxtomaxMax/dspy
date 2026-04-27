from __future__ import annotations

import argparse

from rag_baseline.config import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from rag_baseline.jina_embedder import JinaCodeEmbedder
from rag_baseline.retriever import DenseCodeRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the dense code index with a natural-language query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    retriever = DenseCodeRetriever(
        index_dir=args.index_dir,
        embedder=JinaCodeEmbedder(model_name=args.model_name, device=args.device),
    )
    hits = retriever.search(args.query, top_k=args.top_k)
    print(retriever.format_hits(hits))


if __name__ == "__main__":
    main()
