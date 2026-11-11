"""Interfaz comun de estrategias y utilidades de agrupamiento de oraciones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable, ClassVar, Sequence

import numpy as np

from ..config import CommonConfig, ConfigError
from ..core import Chunk, Document, RawChunk, Span, doc_key
from ..headings import Heading
from ..tokenization import TokenIndex


@dataclass
class DocView:
    """Vista de un documento ya preprocesado, compartida por todas las estrategias:
    las oraciones y sus embeddings se calculan UNA vez."""

    doc: Document
    token_index: TokenIndex
    sentences: list[Span]
    sentence_tokens: np.ndarray  # int64, tokens por oracion
    headings: list[Heading]
    strict_headings: bool
    _embed_fn: Callable[[], np.ndarray] | None = None
    _emb: np.ndarray | None = field(default=None, repr=False)

    def embeddings(self) -> np.ndarray:
        if self._emb is None:
            if self._embed_fn is None:
                raise RuntimeError("Esta estrategia requiere embeddings pero no hay embedder")
            self._emb = np.asarray(self._embed_fn(), dtype=np.float64)
        return self._emb


class Strategy(ABC):
    kind: ClassVar[str]
    needs_embeddings: ClassVar[bool] = False
    #: 'exact' => cada caracter no blanco en exactamente un chunk;
    #: 'tokens' => ventanas de tokens (cobertura total, con posible solapamiento)
    coverage_mode: ClassVar[str] = "exact"
    DEFAULTS: ClassVar[dict[str, Any]] = {}

    def __init__(self, name: str, params: dict[str, Any], common: CommonConfig, seed: int):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ConfigError(f"Parametros desconocidos para '{name}' ({self.kind}): {sorted(unknown)}")
        self.name = name
        self.common = common
        self.seed = seed
        self.params = self._check({**self.DEFAULTS, **params})

    @abstractmethod
    def _check(self, params: dict[str, Any]) -> dict[str, Any]:
        """Valida y normaliza parametros."""

    @property
    def allows_overlap(self) -> bool:
        return False

    @abstractmethod
    def chunk(self, view: DocView) -> list[RawChunk]: ...


# ---------------------------------------------------------------- utilidades


def spans_for_indices(sentences: Sequence[Span], idxs: Sequence[int]) -> tuple[Span, ...]:
    """Spans (uno por corrida de oraciones consecutivas) de un grupo de oraciones."""
    spans: list[Span] = []
    run_start = prev = idxs[0]
    for k in idxs[1:]:
        if k != prev + 1:
            spans.append(Span(sentences[run_start].start, sentences[prev].end))
            run_start = k
        prev = k
    spans.append(Span(sentences[run_start].start, sentences[prev].end))
    return tuple(spans)


def raw_from_indices(view: DocView, idxs: Sequence[int], meta: dict[str, Any] | None = None) -> RawChunk:
    idxs = sorted(int(i) for i in idxs)
    return RawChunk(spans_for_indices(view.sentences, idxs), len(idxs), dict(meta or {}))


def balanced_split(idxs: Sequence[int], tokens: np.ndarray, cap: int) -> list[list[int]]:
    """Divide un grupo ordenado de oraciones en k piezas de tamano similar, cada una
    <= cap tokens (una oracion sola nunca excede cap por la atomizacion previa)."""
    total = int(sum(int(tokens[i]) for i in idxs))
    if total <= cap:
        return [list(idxs)]
    k = ceil(total / cap)
    target = ceil(total / k)
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_t = 0
    for i in idxs:
        t = int(tokens[i])
        if cur and cur_t + t > target:
            groups.append(cur)
            cur, cur_t = [], 0
        cur.append(i)
        cur_t += t
    if cur:
        groups.append(cur)
    return groups


def finalize(view: DocView, raws: Sequence[RawChunk], strategy: str) -> list[Chunk]:
    """Materializa texto, tokens e identificadores. Ordena por posicion de inicio."""
    text = view.doc.text
    key = doc_key(view.doc.doc_id)
    ordered = sorted(raws, key=lambda r: (r.spans[0].start, r.spans[-1].end))
    chunks: list[Chunk] = []
    for idx, r in enumerate(ordered):
        ctext = " ".join(text[s.start : s.end] for s in r.spans)
        ntok = sum(view.token_index.count(s.start, s.end) for s in r.spans)
        chunks.append(
            Chunk(
                chunk_id=f"{strategy}:{key}:{idx:05d}",
                doc_id=view.doc.doc_id,
                strategy=strategy,
                index=idx,
                spans=r.spans,
                text=ctext,
                n_tokens=ntok,
                n_sentences=r.n_sentences,
                meta=r.meta,
            )
        )
    return chunks
