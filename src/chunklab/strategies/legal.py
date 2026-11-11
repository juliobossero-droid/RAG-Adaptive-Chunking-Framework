"""Segmentacion estructural juridica.

Secciones: preambulo / AUTOS Y VISTOS / CONSIDERANDO / RESUELVO (ver headings.py).
  * Una seccion con <= max_tokens se conserva COMPLETA como un unico chunk.
  * Una seccion mas larga se subdivide con la optimizacion semantica exacta (DP).
    Si la seccion es 'resuelvo' se marca `resuelvo_overflow` (el tope de tokens
    impide conservarla integra; hay que contar cuantas veces ocurre).
  * Secciones de menos de `min_section_tokens` (p. ej. un encabezado seguido de otro
    encabezado) se fusionan con la siguiente.
  * Documento sin encabezados detectables: se aplica la DP sobre todo el documento y
    cada chunk lleva `fallback: true` (la 'estructura detectada' es en si misma una
    senal de percepcion util para el agente).
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from ..core import RawChunk, Span
from ..headings import Section, build_sections
from .base import DocView, Strategy, raw_from_indices
from .semantic import check_dp_params, partition_dp


class LegalStructuralStrategy(Strategy):
    kind = "legal_structural"
    needs_embeddings = True
    DEFAULTS = {
        "min_sentences": 3,
        "max_sentences": 15,
        "ideal_sentences": 9,
        "lam": 0.4,
        "objective": "sentence_weighted",
        "undersize_cost": 1000.0,
        "min_section_tokens": 12,
    }

    def _check(self, p: dict[str, Any]) -> dict[str, Any]:
        out = check_dp_params(self.name, p)
        out["min_section_tokens"] = int(p["min_section_tokens"])
        return out

    def _dp_args(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if k != "min_section_tokens"}

    def _merge_tiny(self, view: DocView, sections: list[Section]) -> list[Section]:
        ti, minimum = view.token_index, self.params["min_section_tokens"]
        secs = list(sections)
        i = 0
        while i < len(secs) and len(secs) > 1:
            s = secs[i]
            if ti.count(s.span.start, s.span.end) >= minimum:
                i += 1
                continue
            j = i + 1 if i + 1 < len(secs) else i - 1  # siguiente; si es la ultima, anterior
            a, b = (s, secs[j]) if j > i else (secs[j], s)
            label = secs[j].label if j > i else secs[j].label
            merged = Section(label, Span(a.span.start, b.span.end))
            lo = min(i, j)
            secs[lo : lo + 2] = [merged]
            i = max(lo - 1, 0)
        return secs

    def chunk(self, view: DocView) -> list[RawChunk]:
        n = len(view.sentences)
        if n == 0:
            return []
        emb, tokens, cap = view.embeddings(), view.sentence_tokens, self.common.max_tokens
        sections = self._merge_tiny(view, build_sections(view.doc.text, view.headings))
        if not sections:
            return self._dp_range(view, 0, n, {"section": "documento", "fallback": True})
        starts = [s.start for s in view.sentences]
        out: list[RawChunk] = []
        for sec in sections:
            lo = bisect_left(starts, sec.span.start)
            hi = bisect_left(starts, sec.span.end)
            if hi <= lo:
                continue
            total = int(tokens[lo:hi].sum())
            if total <= cap:
                out.append(raw_from_indices(view, range(lo, hi), {"section": sec.label, "section_whole": True}))
            else:
                meta: dict[str, Any] = {"section": sec.label, "section_whole": False}
                if sec.label == "resuelvo":
                    meta["resuelvo_overflow"] = True
                out.extend(self._dp_range(view, lo, hi, meta))
        return out

    def _dp_range(self, view: DocView, lo: int, hi: int, meta: dict[str, Any]) -> list[RawChunk]:
        parts, _ = partition_dp(
            view.embeddings()[lo:hi],
            view.sentence_tokens[lo:hi],
            max_tokens=self.common.max_tokens,
            **self._dp_args(),
        )
        return [raw_from_indices(view, range(lo + i, lo + j), dict(meta)) for i, j in parts]
