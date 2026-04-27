from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from grep_agent.agent import GrepCodeAgent, configure_dspy_lm
from grep_agent.config import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_REACT_MAX_ITERS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_TOOL_CHAR_LIMIT,
    DEFAULT_TOOL_MAX_MATCHES,
    DEFAULT_TOOL_MAX_READ_LINES,
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    REPO_ROOT,
)
from rag_baseline.benchmark_utils import load_benchmark
from rag_baseline.io_utils import ensure_parent_dir, write_json, write_jsonl
from rag_baseline.types import AnswerRunRecord


def _extract_tool_usage(trajectory: dict) -> tuple[list[str], list[str], int]:
    used_chunk_ids: list[str] = []
    used_files: list[str] = []
    tool_calls = 0

    for key, value in trajectory.items():
        if not key.startswith("observation_") or not isinstance(value, str):
            continue
        tool_calls += 1
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        file_path = payload.get("file_path")
        if file_path:
            used_files.append(str(file_path))

        for item in payload.get("results", []):
            if isinstance(item, dict) and item.get("file_path"):
                used_files.append(str(item["file_path"]))

    return sorted(set(used_chunk_ids)), sorted(set(used_files)), tool_calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grep-based code agent on the benchmark.")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--lm-model", required=True)
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-iters", type=int, default=DEFAULT_REACT_MAX_ITERS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tool-char-limit", type=int, default=DEFAULT_TOOL_CHAR_LIMIT)
    parser.add_argument("--tool-max-matches", type=int, default=DEFAULT_TOOL_MAX_MATCHES)
    parser.add_argument("--tool-max-read-lines", type=int, default=DEFAULT_TOOL_MAX_READ_LINES)
    parser.add_argument("--tool-timeout-seconds", type=int, default=DEFAULT_TOOL_TIMEOUT_SECONDS)
    parser.add_argument("--output-path", default=str(DEFAULT_RESULTS_DIR / "grep_agent_predictions.jsonl"))
    parser.add_argument("--summary-path", default=str(DEFAULT_RESULTS_DIR / "grep_agent_run_summary.json"))
    args = parser.parse_args()

    configure_dspy_lm(
        lm_model=args.lm_model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        track_usage=True,
    )

    agent = GrepCodeAgent(
        repo_root=Path(args.repo_root),
        source_dir=Path(args.source_dir),
        max_iters=args.max_iters,
        char_limit=args.tool_char_limit,
        max_matches=args.tool_max_matches,
        max_read_lines=args.tool_max_read_lines,
        timeout_seconds=args.tool_timeout_seconds,
    )
    examples = load_benchmark(Path(args.benchmark_path))
    if args.limit is not None:
        examples = examples[: args.limit]

    records: list[AnswerRunRecord] = []
    for idx, example in enumerate(examples, start=1):
        import dspy

        start = time.perf_counter()
        with dspy.track_usage() as usage_tracker:
            pred = agent.answer(example.question)
        elapsed = time.perf_counter() - start

        trajectory = getattr(pred, "trajectory", {}) or {}
        used_chunk_ids, used_files, tool_calls = _extract_tool_usage(trajectory)
        token_usage = usage_tracker.get_total_tokens()

        records.append(
            AnswerRunRecord(
                question=example.question,
                gold_answer=example.answer,
                pred_answer=getattr(pred, "answer", ""),
                hops=example.hops,
                retrieval_path=example.retrieval_path,
                trajectory=trajectory,
                tool_calls=tool_calls,
                elapsed_seconds=elapsed,
                token_usage=token_usage,
                used_chunk_ids=used_chunk_ids,
                used_files=used_files,
            )
        )
        print(f"[{idx}/{len(examples)}] completed")

    rows = [record.to_dict() for record in records]
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    ensure_parent_dir(output_path)
    ensure_parent_dir(summary_path)
    write_jsonl(output_path, rows)
    write_json(
        summary_path,
        {
            "num_examples": len(records),
            "benchmark_path": args.benchmark_path,
            "repo_root": args.repo_root,
            "source_dir": args.source_dir,
            "lm_model": args.lm_model,
            "max_iters": args.max_iters,
            "temperature": args.temperature,
            "tool_char_limit": args.tool_char_limit,
            "tool_max_matches": args.tool_max_matches,
            "tool_max_read_lines": args.tool_max_read_lines,
            "tool_timeout_seconds": args.tool_timeout_seconds,
            "output_path": str(output_path),
        },
    )


if __name__ == "__main__":
    main()
