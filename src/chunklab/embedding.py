"""Embeddings de oraciones (para las estrategias semanticas) y su cache en disco."""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .core import Document, Span


class Embedder(Protocol):
    backend: str
    model_name: str
    revision: str | None
    prefix: str
    dim: int
    effective_token_limit: int  # max_seq_length - tokens especiales - tokens del prefijo

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Matriz (n, dim), float32, filas normalizadas L2."""

    def fingerprint(self) -> str: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        prefix: str = "",
        revision: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.backend = "sentence-transformers"
        self.model_name = model_name
        self.revision = revision
        self.prefix = prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)
        tok = self.model.tokenizer
        specials = int(tok.num_special_tokens_to_add(pair=False))
        prefix_tokens = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        self.effective_token_limit = self.max_seq_length - specials - prefix_tokens

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        arr = self.model.encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)

    def fingerprint(self) -> str:
        return f"{self.backend}|{self.model_name}|{self.revision}|{self.prefix}|{self.dim}"


class HashingEmbedder:
    """Embedder determinista por bolsa de palabras con hashing. SOLO para pruebas y
    simulacros del pipeline; no tiene validez cientifica."""

    def __init__(self, dim: int = 128, effective_token_limit: int = 100_000):
        self.backend = "hashing"
        self.model_name = f"hashing-{dim}"
        self.revision = None
        self.prefix = ""
        self.dim = dim
        self.effective_token_limit = effective_token_limit

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                h = zlib.crc32(w.encode("utf-8"))
                out[r, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            norm = float(np.linalg.norm(out[r]))
            if norm == 0.0:
                out[r, 0] = 1.0
            else:
                out[r] /= norm
        return out

    def fingerprint(self) -> str:
        return f"hashing|{self.dim}"


def spans_digest(spans: Sequence[Span]) -> str:
    h = hashlib.sha256()
    for s in spans:
        h.update(f"{s.start}-{s.end};".encode())
    return h.hexdigest()


class EmbeddingCache:
    """Cache de embeddings de oraciones. La clave depende del documento, de la
    segmentacion exacta en oraciones y del modelo; si algo cambia, no se reutiliza."""

    def __init__(self, root: Path, embedder: Embedder):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder

    def key(self, doc: Document, sentences: Sequence[Span]) -> str:
        payload = json.dumps(
            {"doc": doc.sha256, "emb": self.embedder.fingerprint(), "sent": spans_digest(sentences)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, doc: Document, sentences: Sequence[Span], texts: Sequence[str]) -> np.ndarray:
        k = self.key(doc, sentences)
        path = self.root / f"{k}.npy"
        if path.exists():
            try:
                arr = np.load(path)
                if arr.shape == (len(sentences), self.embedder.dim):
                    return arr
            except (OSError, ValueError):
                pass  # cache corrupto: se recalcula
        arr = self.embedder.encode(texts)
        norms = np.linalg.norm(arr, axis=1)
        if len(arr) and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("El embedder devolvio vectores no normalizados")
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(path)
        return arr
