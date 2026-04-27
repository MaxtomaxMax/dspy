from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from rag_baseline.benchmark_utils import extract_gold_files
from rag_baseline.io_utils import load_jsonl, write_json
from rag_baseline.metrics import exact_match, f1_score


def evaluate_answer_file(predictions_path: Path) -> dict:
    rows = load_jsonl(predictions_path)

    em_scores: list[float] = []
    f1_scores: list[float] = []
    tool_calls: list[int] = []
    latencies: list[float] = []
    coverage_by_hops: dict[int, list[float]] = defaultdict(list)

    for row in rows:
        pred_answer = row.get("pred_answer", "")
        gold_answer = row.get("gold_answer", "")
        em_scores.append(1.0 if exact_match(pred_answer, gold_answer) else 0.0)
        f1_scores.append(f1_score(pred_answer, gold_answer))
        tool_calls.append(int(row.get("tool_calls", 0)))
        latencies.append(float(row.get("elapsed_seconds", 0.0)))

        gold_files = extract_gold_files(row.get("retrieval_path", []))
        used_files = set(row.get("used_files", []))
        coverage = len(gold_files.intersection(used_files)) / max(len(gold_files), 1)
        coverage_by_hops[int(row.get("hops", 0))].append(coverage)

    return {
        "num_examples": len(rows),
        "answer_em": sum(em_scores) / max(len(em_scores), 1),
        "answer_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "avg_tool_calls": sum(tool_calls) / max(len(tool_calls), 1),
        "avg_latency_seconds": sum(latencies) / max(len(latencies), 1),
        "trajectory_chain_coverage_by_hops": {
            str(hops): sum(values) / max(len(values), 1) for hops, values in sorted(coverage_by_hops.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense RAG answer outputs with local EM/F1 metrics.")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    summary = evaluate_answer_file(Path(args.predictions_path))
    if args.output_path:
        write_json(Path(args.output_path), summary)
    print(summary)


if __name__ == "__main__":
    main()
