from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from rag_baseline.types import CodeChunk


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _build_chunk_id(file_path: str, symbol_path: str, start_line: int, end_line: int) -> str:
    raw = f"{file_path}:{symbol_path}:{start_line}:{end_line}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{Path(file_path).stem}-{digest}"


def _render_signature(node: ast.AST, qualified_name: str) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(_safe_unparse(base) for base in node.bases) or "object"
        return f"class {qualified_name}({bases})"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = _safe_unparse(node.args)
        return f"{prefix} {qualified_name}({args})"
    return qualified_name


def _compose_chunk_text(
    file_path: str,
    chunk_type: str,
    symbol_path: str,
    signature: str,
    docstring: str,
    code: str,
) -> str:
    parts = [
        f"file_path: {file_path}",
        f"chunk_type: {chunk_type}",
        f"symbol_path: {symbol_path or '<module>'}",
        f"signature: {signature or '<none>'}",
    ]
    if docstring:
        parts.append(f"docstring:\n{docstring}")
    parts.append(f"code:\n{code}")
    return "\n\n".join(parts)


def _make_chunk(
    *,
    repo_root: Path,
    file_path: Path,
    source_lines: list[str],
    node: ast.AST,
    chunk_type: str,
    symbol_path: str,
) -> CodeChunk | None:
    start_line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    if start_line is None or end_line is None:
        return None

    code = "\n".join(source_lines[start_line - 1 : end_line])
    relative_path = file_path.relative_to(repo_root).as_posix()
    signature = _render_signature(node, symbol_path.split(".")[-1] if symbol_path else file_path.stem)
    docstring = ast.get_docstring(node) or ""
    text = _compose_chunk_text(relative_path, chunk_type, symbol_path, signature, docstring, code)
    chunk_id = _build_chunk_id(relative_path, symbol_path, start_line, end_line)

    return CodeChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        file_path=relative_path,
        symbol_path=symbol_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring,
        text=text,
        code=code,
    )


def _make_file_overview(repo_root: Path, file_path: Path, source: str, tree: ast.Module) -> CodeChunk:
    relative_path = file_path.relative_to(repo_root).as_posix()
    imports: list[str] = []
    top_level_defs: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported = ", ".join(alias.name for alias in node.names)
            imports.append(f"{module}: {imported}")
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_defs.append(node.name)

    summary_lines = [
        f"file_path: {relative_path}",
        "chunk_type: file_overview",
        f"top_level_definitions: {', '.join(top_level_defs) or '<none>'}",
        f"imports: {'; '.join(imports) or '<none>'}",
        "code:",
        source,
    ]
    text = "\n".join(summary_lines)
    total_lines = len(source.splitlines())

    return CodeChunk(
        chunk_id=_build_chunk_id(relative_path, "<file_overview>", 1, total_lines),
        chunk_type="file_overview",
        file_path=relative_path,
        symbol_path="",
        start_line=1,
        end_line=total_lines,
        signature=f"module {relative_path}",
        docstring=ast.get_docstring(tree) or "",
        text=text,
        code=source,
        metadata={
            "imports": imports,
            "top_level_definitions": top_level_defs,
        },
    )


def extract_code_chunks(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source)

    chunks: list[CodeChunk] = [_make_file_overview(repo_root, file_path, source, tree)]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_chunk = _make_chunk(
                repo_root=repo_root,
                file_path=file_path,
                source_lines=source_lines,
                node=node,
                chunk_type="class",
                symbol_path=node.name,
            )
            if class_chunk is not None:
                chunks.append(class_chunk)

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = _make_chunk(
                        repo_root=repo_root,
                        file_path=file_path,
                        source_lines=source_lines,
                        node=child,
                        chunk_type="method",
                        symbol_path=f"{node.name}.{child.name}",
                    )
                    if method_chunk is not None:
                        chunks.append(method_chunk)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_chunk = _make_chunk(
                repo_root=repo_root,
                file_path=file_path,
                source_lines=source_lines,
                node=node,
                chunk_type="function",
                symbol_path=node.name,
            )
            if func_chunk is not None:
                chunks.append(func_chunk)

    return chunks


def build_corpus_from_glob(repo_root: Path, pattern: str) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for file_path in sorted(repo_root.glob(pattern)):
        if file_path.is_file():
            chunks.extend(extract_code_chunks(file_path, repo_root))
    return chunks
