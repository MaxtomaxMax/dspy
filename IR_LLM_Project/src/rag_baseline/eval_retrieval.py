from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rag_baseline.benchmark_utils import extract_gold_files, load_benchmark
from rag_baseline.config import DEFAULT_BENCHMARK_PATH, DEFAULT_INDEX_DIR
from rag_baseline.io_utils import write_json
from rag_baseline.jina_embedder import JinaCodeEmbedder
from rag_baseline.retriever import DenseCodeRetriever


def evaluate_retrieval(
    *,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    top_k: int = 8,
    output_path: Path | None = None,
) -> dict:
    examples = load_benchmark(benchmark_path)

    retriever = DenseCodeRetriever(index_dir=index_dir, embedder=JinaCodeEmbedder())

    hit_count = 0
    reciprocal_ranks: list[float] = []
    coverage_by_hops: dict[int, list[float]] = defaultdict(list)
    per_example: list[dict] = []

    for example in examples:
        gold_files = extract_gold_files(example.retrieval_path)
        hits = retriever.search(example.question, top_k=top_k)
        retrieved_files = [hit.file_path for hit in hits]

        rank = 0
        for idx, file_path in enumerate(retrieved_files, start=1):
            if file_path in gold_files:
                rank = idx
                break

        if rank > 0:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        covered = len(gold_files.intersection(retrieved_files))
        coverage = covered / max(len(gold_files), 1)
        coverage_by_hops[example.hops].append(coverage)
        per_example.append(
            {
                "question": example.question,
                "gold_files": sorted(gold_files),
                "retrieved_files": retrieved_files,
                "first_hit_rank": rank or None,
                "chain_coverage": coverage,
            }
        )

    file_recall = hit_count / max(len(examples), 1)
    mrr = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
    hop_coverage = {
        str(hops): sum(values) / max(len(values), 1) for hops, values in sorted(coverage_by_hops.items())
    }

    return {
        "num_examples": len(examples),
        "top_k": top_k,
        "file_recall_at_k": file_recall,
        "mrr": mrr,
        "chain_coverage_by_hops": hop_coverage,
    }
    if output_path is not None:
        write_json(output_path, {"summary": summary, "per_example": per_example})
    return summary


if __name__ == "__main__":
    metrics = evaluate_retrieval()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
