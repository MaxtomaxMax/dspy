from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from grep_agent.config import (
    DEFAULT_SOURCE_DIR,
    DEFAULT_TOOL_CHAR_LIMIT,
    DEFAULT_TOOL_MAX_MATCHES,
    DEFAULT_TOOL_MAX_READ_LINES,
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    REPO_ROOT,
)


class GrepCodeSearchToolkit:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        source_dir: Path = DEFAULT_SOURCE_DIR,
        char_limit: int = DEFAULT_TOOL_CHAR_LIMIT,
        max_matches: int = DEFAULT_TOOL_MAX_MATCHES,
        max_read_lines: int = DEFAULT_TOOL_MAX_READ_LINES,
        timeout_seconds: int = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.source_dir = Path(source_dir).resolve()
        self.char_limit = char_limit
        self.max_matches = max_matches
        self.max_read_lines = max_read_lines
        self.timeout_seconds = timeout_seconds

        if not self.source_dir.is_relative_to(self.repo_root):
            raise ValueError(f"source_dir must stay inside repo_root: {self.source_dir}")

        self._search_root_arg = str(self.source_dir.relative_to(self.repo_root))
        self.reset_session()

    def reset_session(self) -> None:
        self._last_call_signature: tuple[Any, ...] | None = None

    def global_search(self, regex_pattern: str) -> str:
        pattern = regex_pattern.strip()
        if not pattern:
            return self._error_payload("regex_pattern cannot be empty.")

        duplicate_error = self._guard_repeated_call(("global_search", pattern))
        if duplicate_error is not None:
            return duplicate_error

        command = [
            "rg",
            "--json",
            "-n",
            "-S",
            "--glob",
            "*.py",
            "-e",
            pattern,
            self._search_root_arg,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return self._error_payload("ripgrep (rg) is not installed or not available in PATH.")
        except subprocess.TimeoutExpired:
            return self._error_payload("global_search timed out. Narrow the regex pattern.")

        if completed.returncode not in (0, 1):
            message = completed.stderr.strip() or f"rg exited with code {completed.returncode}."
            return self._error_payload(f"global_search failed: {message}")

        results: list[dict[str, Any]] = []
        unique_files: set[str] = set()
        total_matches = 0

        for raw_line in completed.stdout.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue

            data = event.get("data", {})
            file_path = self._normalize_repo_path(self._extract_json_text(data.get("path", {})))
            line_number = data.get("line_number")
            line_text = self._extract_json_text(data.get("lines", {})).strip()
            if not file_path:
                continue

            total_matches += 1
            unique_files.add(file_path)
            if len(results) < self.max_matches:
                results.append(
                    {
                        "file_path": file_path,
                        "line_number": line_number,
                        "line_text": line_text[:180],
                    }
                )

        message = f"Found {total_matches} matches across {len(unique_files)} files."
        if total_matches == 0:
            message = "No matches found. Refine the regex pattern or search for a more stable symbol name."
        elif total_matches > len(results):
            message += " Output was truncated; refine the regex pattern before reading files."

        payload = {
            "status": "ok",
            "tool": "global_search",
            "pattern": pattern,
            "total_matches": total_matches,
            "returned_matches": len(results),
            "files": sorted(unique_files),
            "results": results,
            "message": message,
            "truncated": total_matches > len(results),
        }
        return self._finalize_payload(payload)

    def read_file_lines(self, filepath: str, start_line: int, end_line: int) -> str:
        normalized_path = filepath.strip()
        if not normalized_path:
            return self._error_payload("filepath cannot be empty.")
        if start_line <= 0 or end_line <= 0:
            return self._error_payload("start_line and end_line must be positive integers.")
        if end_line < start_line:
            return self._error_payload("end_line must be greater than or equal to start_line.")

        duplicate_error = self._guard_repeated_call(("read_file_lines", normalized_path, start_line, end_line))
        if duplicate_error is not None:
            return duplicate_error

        resolved_path = self._resolve_read_path(normalized_path)
        if resolved_path is None:
            return self._error_payload(
                "filepath must point to a Python file inside the indexed source tree under dspy/."
            )
        if not resolved_path.exists():
            return self._error_payload(f"File does not exist: {normalized_path}")

        requested_line_count = end_line - start_line + 1
        effective_end_line = end_line
        truncated_by_range = False
        if requested_line_count > self.max_read_lines:
            effective_end_line = start_line + self.max_read_lines - 1
            truncated_by_range = True

        lines = resolved_path.read_text(encoding="utf-8", errors="replace").splitlines()
        actual_start = min(start_line, len(lines) + 1)
        actual_end = min(effective_end_line, len(lines))

        selected_lines: list[str] = []
        if actual_start <= actual_end:
            for line_number in range(actual_start, actual_end + 1):
                selected_lines.append(f"{line_number}: {lines[line_number - 1]}")

        file_path = str(resolved_path.relative_to(self.repo_root))
        message = f"Read lines {actual_start}-{actual_end} from {file_path}."
        if truncated_by_range:
            message += f" Requested range was clamped to {self.max_read_lines} lines."
        if actual_start > len(lines):
            message = f"Requested range starts after EOF. {file_path} has only {len(lines)} lines."

        payload = {
            "status": "ok",
            "tool": "read_file_lines",
            "file_path": file_path,
            "requested_start_line": start_line,
            "requested_end_line": end_line,
            "start_line": actual_start,
            "end_line": actual_end,
            "line_count": len(selected_lines),
            "content": "\n".join(selected_lines),
            "message": message,
            "truncated": truncated_by_range,
        }
        return self._finalize_payload(payload)

    def _guard_repeated_call(self, signature: tuple[Any, ...]) -> str | None:
        if self._last_call_signature == signature:
            return self._error_payload(
                "Repeated identical tool call blocked to prevent a ReAct loop. "
                "Change the regex pattern or request a different file range."
            )
        self._last_call_signature = signature
        return None

    def _resolve_read_path(self, raw_path: str) -> Path | None:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.repo_root / candidate).resolve()
        if not resolved.is_relative_to(self.source_dir):
            return None
        if resolved.suffix != ".py":
            return None
        return resolved

    def _normalize_repo_path(self, raw_path: str) -> str:
        if not raw_path:
            return ""
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.repo_root / candidate).resolve()
        try:
            return str(resolved.relative_to(self.repo_root))
        except ValueError:
            return str(candidate).replace("\\", "/")

    @staticmethod
    def _extract_json_text(node: Any) -> str:
        if isinstance(node, dict):
            return str(node.get("text", ""))
        return str(node or "")

    def _error_payload(self, message: str) -> str:
        return self._finalize_payload({"status": "error", "message": message, "truncated": False})

    def _finalize_payload(self, payload: dict[str, Any]) -> str:
        trimmed = dict(payload)

        if "results" in trimmed and isinstance(trimmed["results"], list):
            original_results = list(trimmed["results"])
            results = original_results[:]
            while results:
                trimmed["results"] = results
                rendered = json.dumps(trimmed, ensure_ascii=False, indent=2)
                if len(rendered) <= self.char_limit:
                    break
                results = results[:-1]
            else:
                trimmed["results"] = []

            if len(trimmed["results"]) < len(original_results):
                trimmed["truncated"] = True
                trimmed["message"] = (
                    f"{trimmed.get('message', 'Result truncated.')} "
                    "Too many matches were returned; refine the regex pattern."
                )

        if "content" in trimmed and isinstance(trimmed["content"], str):
            original_content = trimmed["content"]
            content = original_content
            while content:
                trimmed["content"] = content
                rendered = json.dumps(trimmed, ensure_ascii=False, indent=2)
                if len(rendered) <= self.char_limit:
                    break
                new_length = max(120, int(len(content) * 0.8))
                if new_length >= len(content):
                    new_length = len(content) - 1
                content = content[:new_length].rstrip()
                if not content.endswith("...[truncated]..."):
                    content += "\n...[truncated]..."

            if trimmed.get("content") != original_content:
                trimmed["truncated"] = True
                trimmed["message"] = (
                    f"{trimmed.get('message', 'Result truncated.')} "
                    "Read a smaller line range if you need the omitted content."
                )

        rendered = json.dumps(trimmed, ensure_ascii=False, indent=2)
        if len(rendered) > self.char_limit:
            fallback = {
                "status": trimmed.get("status", "error"),
                "message": "Result exceeded the tool output limit. Narrow the query or read a smaller line range.",
                "truncated": True,
            }
            return json.dumps(fallback, ensure_ascii=False, indent=2)
        return rendered
