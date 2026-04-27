from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_baseline.config import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, DEFAULT_SOURCE_GLOB, REPO_ROOT
from rag_baseline.indexer import build_and_save_index
from rag_baseline.jina_embedder import JinaCodeEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dense RAG index for the DSPy codebase.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--output-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=None)
    args = parser.parse_args()

    embedder = JinaCodeEmbedder(
        model_name=args.model_name,
        device=args.device,
        use_fp16=args.use_fp16,
        max_seq_length=args.max_seq_length,
    )
    metadata = build_and_save_index(
        repo_root=Path(args.repo_root),
        source_glob=args.source_glob,
        output_dir=Path(args.output_dir),
        embedder=embedder,
        batch_size=args.batch_size,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
