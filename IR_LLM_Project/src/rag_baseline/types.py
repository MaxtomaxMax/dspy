from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CodeChunk:
    chunk_id: str
    chunk_type: str
    file_path: str
    symbol_path: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    text: str
    code: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        return cls(**data)


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    file_path: str
    symbol_path: str
    chunk_type: str
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkExample:
    question: str
    answer: str
    retrieval_path: list[str]
    hops: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkExample":
        return cls(
            question=data["question"],
            answer=data["answer"],
            retrieval_path=data.get("retrieval_path", []),
            hops=data.get("hops", 0),
        )


@dataclass
class AnswerRunRecord:
    question: str
    gold_answer: str
    pred_answer: str
    hops: int
    retrieval_path: list[str]
    trajectory: dict[str, Any]
    tool_calls: int
    elapsed_seconds: float
    token_usage: dict[str, Any]
    used_chunk_ids: list[str]
    used_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
