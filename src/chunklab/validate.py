"""Validacion estricta de los chunks de un documento. Un error NUNCA se ignora."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import Chunk, Document
from .tokenization import TokenizerAdapter


@dataclass(frozen=True)
class Issue:
    severity: str  # 'error' | 'warning'
    code: str
    message: str
    doc_id: str
    strategy: str

    def to_dict(self) -> dict[str, str]:
        return vars(self).copy()


def validate_chunks(
    doc: Document,
    chunks: Sequence[Chunk],
    *,
    strategy: str,
    max_tokens: int,
    hard_limit: int,
    tokenizer: TokenizerAdapter,
    coverage_mode: str,
    allows_overlap: bool,
) -> list[Issue]:
    text = doc.text
    issues: list[Issue] = []

    def add(sev: str, code: str, msg: str) -> None:
        issues.append(Issue(sev, code, msg, doc.doc_id, strategy))

    if text.strip() and not chunks:
        add("error", "NO_CHUNKS", "el documento no genero chunks")
        return issues
    if len({c.chunk_id for c in chunks}) != len(chunks):
        add("error", "DUP_ID", "chunk_id duplicados")
    cover = np.zeros(len(text), dtype=np.int32)
    prev_start = -1
    for c in chunks:
        if c.spans[0].start < prev_start:
            add("error", "ORDER", f"chunk {c.index} fuera de orden")
        prev_start = c.spans[0].start
        last_end = -1
        for s in c.spans:
            if s.end > len(text) or s.start < 0:
                add("error", "RANGE", f"chunk {c.index}: span fuera de rango")
                continue
            if s.start < last_end:
                add("error", "SPAN_ORDER", f"chunk {c.index}: spans solapados o desordenados")
            last_end = s.end
            cover[s.start : s.end] += 1
        expected = " ".join(text[s.start : s.end] for s in c.spans)
        if c.text != expected:
            add("error", "TEXT_MISMATCH", f"chunk {c.index}: el texto no coincide con sus spans")
        if c.n_tokens > max_tokens:
            add("error", "TOKEN_CAP", f"chunk {c.index}: {c.n_tokens} tokens > {max_tokens}")
        real = tokenizer.count(c.text)
        if real > hard_limit:
            add("error", "HARD_LIMIT", f"chunk {c.index}: {real} tokens reales > limite del modelo {hard_limit}")
        elif real > max_tokens:
            add("warning", "TOKEN_DRIFT", f"chunk {c.index}: {real} tokens reales > max_tokens={max_tokens}")
    nonws = np.fromiter((not ch.isspace() for ch in text), dtype=bool, count=len(text))
    total = int(nonws.sum())
    uncovered = int((nonws & (cover == 0)).sum())
    duplicated = int((nonws & (cover > 1)).sum())
    if coverage_mode == "exact":
        if uncovered:
            add("error", "COVERAGE", f"{uncovered} caracteres sin cubrir")
        if duplicated:
            add("error", "DUPLICATION", f"{duplicated} caracteres en mas de un chunk")
    else:
        if total and uncovered / total > 0.001:
            add("error", "COVERAGE", f"{uncovered} caracteres sin cubrir ({uncovered / total:.2%})")
        elif uncovered:
            add("warning", "COVERAGE", f"{uncovered} caracteres sin cubrir (descartados por el tokenizador)")
        if duplicated and not allows_overlap:
            add("error", "DUPLICATION", f"{duplicated} caracteres duplicados sin solapamiento declarado")
    return issues
