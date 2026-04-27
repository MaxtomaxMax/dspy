from __future__ import annotations

import argparse
from pathlib import Path

from grep_agent.config import (
    DEFAULT_SOURCE_DIR,
    DEFAULT_TOOL_CHAR_LIMIT,
    DEFAULT_TOOL_MAX_MATCHES,
    DEFAULT_TOOL_MAX_READ_LINES,
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    REPO_ROOT,
)
from grep_agent.grep_tools import GrepCodeSearchToolkit


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the DSPy codebase with ripgrep-backed tools.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--tool-char-limit", type=int, default=DEFAULT_TOOL_CHAR_LIMIT)
    parser.add_argument("--tool-max-matches", type=int, default=DEFAULT_TOOL_MAX_MATCHES)
    parser.add_argument("--tool-max-read-lines", type=int, default=DEFAULT_TOOL_MAX_READ_LINES)
    parser.add_argument("--tool-timeout-seconds", type=int, default=DEFAULT_TOOL_TIMEOUT_SECONDS)

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Run a regex search.")
    search_parser.add_argument("--pattern", required=True)

    read_parser = subparsers.add_parser("read", help="Read a file line range.")
    read_parser.add_argument("--file-path", required=True)
    read_parser.add_argument("--start-line", required=True, type=int)
    read_parser.add_argument("--end-line", required=True, type=int)

    args = parser.parse_args()

    toolkit = GrepCodeSearchToolkit(
        repo_root=Path(args.repo_root),
        source_dir=Path(args.source_dir),
        char_limit=args.tool_char_limit,
        max_matches=args.tool_max_matches,
        max_read_lines=args.tool_max_read_lines,
        timeout_seconds=args.tool_timeout_seconds,
    )

    if args.command == "search":
        print(toolkit.global_search(args.pattern))
        return

    print(toolkit.read_file_lines(args.file_path, args.start_line, args.end_line))


if __name__ == "__main__":
    main()
