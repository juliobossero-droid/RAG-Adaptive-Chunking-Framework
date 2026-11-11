"""Tipos de datos centrales. Todos los offsets son indices de caracter sobre el
texto canonico del documento (ver io.canonicalize)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class Span:
    """Intervalo semiabierto [start, end) sobre el texto canonico."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Span invalido: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: "Span") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str  # texto canonico
    sha256: str  # hash del texto canonico (utf-8)
    source_path: str = ""
    encoding: str = "utf-8"


def doc_key(doc_id: str) -> str:
    """Identificador corto y seguro para nombres de archivo."""
    return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RawChunk:
    """Resultado de una estrategia antes de materializar el texto."""

    spans: tuple[Span, ...]
    n_sentences: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    index: int
    spans: tuple[Span, ...]  # uno solo si el chunk es contiguo
    text: str
    n_tokens: int
    n_sentences: int  # 0 en estrategias por ventana de tokens
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> int:
        return self.spans[0].start

    @property
    def end(self) -> int:
        return self.spans[-1].end

    @property
    def is_contiguous(self) -> bool:
        return len(self.spans) == 1

    def overlap_chars(self, span: Span) -> int:
        """Caracteres de `span` cubiertos por este chunk (para evaluar evidencia)."""
        return sum(s.overlap(span) for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "strategy": self.strategy,
            "index": self.index,
            "spans": [[s.start, s.end] for s in self.spans],
            "text": self.text,
            "n_tokens": self.n_tokens,
            "n_sentences": self.n_sentences,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Chunk":
        return Chunk(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            strategy=d["strategy"],
            index=int(d["index"]),
            spans=tuple(Span(int(a), int(b)) for a, b in d["spans"]),
            text=d["text"],
            n_tokens=int(d["n_tokens"]),
            n_sentences=int(d["n_sentences"]),
            meta=dict(d.get("meta", {})),
        )


class ChunkingError(RuntimeError):
    """Violacion de un invariante de segmentacion (nunca se ignora en silencio)."""
