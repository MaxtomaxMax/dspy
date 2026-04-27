from __future__ import annotations

import re
from pathlib import Path

from rag_baseline.io_utils import load_json
from rag_baseline.types import BenchmarkExample


FILE_PATTERN = re.compile(r"([a-zA-Z0-9_./\\-]+\.py)")


def load_benchmark(path: Path) -> list[BenchmarkExample]:
    return [BenchmarkExample.from_dict(item) for item in load_json(path)]


def extract_gold_files(paths: list[str]) -> set[str]:
    gold_files: set[str] = set()
    for item in paths:
        gold_files.update(match.replace("\\", "/") for match in FILE_PATTERN.findall(item))
    return gold_files
