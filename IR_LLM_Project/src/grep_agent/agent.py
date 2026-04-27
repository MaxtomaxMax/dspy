from __future__ import annotations

from pathlib import Path

from grep_agent.config import (
    DEFAULT_REACT_MAX_ITERS,
    DEFAULT_SOURCE_DIR,
    DEFAULT_TOOL_CHAR_LIMIT,
    DEFAULT_TOOL_MAX_MATCHES,
    DEFAULT_TOOL_MAX_READ_LINES,
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    REPO_ROOT,
)
from grep_agent.grep_tools import GrepCodeSearchToolkit
from rag_baseline.agent import configure_dspy_lm


class GrepCodeAgent:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        source_dir: Path = DEFAULT_SOURCE_DIR,
        max_iters: int = DEFAULT_REACT_MAX_ITERS,
        char_limit: int = DEFAULT_TOOL_CHAR_LIMIT,
        max_matches: int = DEFAULT_TOOL_MAX_MATCHES,
        max_read_lines: int = DEFAULT_TOOL_MAX_READ_LINES,
        timeout_seconds: int = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        import dspy

        self.tools = GrepCodeSearchToolkit(
            repo_root=repo_root,
            source_dir=source_dir,
            char_limit=char_limit,
            max_matches=max_matches,
            max_read_lines=max_read_lines,
            timeout_seconds=timeout_seconds,
        )
        self.react = dspy.ReAct(
            "question -> answer",
            tools=[self.tools.global_search, self.tools.read_file_lines],
            max_iters=max_iters,
        )

    def answer(self, question: str):
        self.tools.reset_session()
        return self.react(question=question)


__all__ = ["GrepCodeAgent", "configure_dspy_lm"]
