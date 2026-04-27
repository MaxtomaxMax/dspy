from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_SOURCE_GLOB = "dspy/**/*.py"
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "DSPy_Multihop_QA.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "artifacts" / "dense_rag_index"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"
DEFAULT_TOP_K = 8
DEFAULT_PREVIEW_CHARS = 1200
DEFAULT_REACT_MAX_ITERS = 6
