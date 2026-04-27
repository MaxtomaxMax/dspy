from __future__ import annotations

from typing import Iterable

import numpy as np

from rag_baseline.config import DEFAULT_MODEL_NAME


class JinaCodeEmbedder:
    """Thin wrapper around jina-code-embeddings-1.5b with asymmetric prompts."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        max_seq_length: int | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.use_fp16 = use_fp16
        self.max_seq_length = max_seq_length
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"trust_remote_code": self.trust_remote_code}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
            if self.use_fp16 and self.device and self.device.startswith("cuda"):
                self._model.half()
        return self._model

    def encode_documents(self, texts: Iterable[str], batch_size: int = 8, normalize_embeddings: bool = True) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            prompt_name="nl2code_document",
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, texts: Iterable[str], batch_size: int = 8, normalize_embeddings: bool = True) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            prompt_name="nl2code_query",
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
