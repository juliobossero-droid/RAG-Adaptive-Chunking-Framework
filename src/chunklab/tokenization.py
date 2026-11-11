"""Tokenizacion con offsets. El presupuesto de tokens de TODAS las estrategias se
mide con el tokenizador del modelo de embeddings, para evitar truncamiento."""

from __future__ import annotations

import re
from bisect import bisect_left
from typing import Any, Protocol, Sequence


class TokenizerAdapter(Protocol):
    name: str
    num_special_tokens: int

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Offsets (start, end) de los tokens que portan caracteres, en orden."""

    def count(self, text: str) -> int:
        """Cantidad real de tokens (sin tokens especiales)."""


class TokenIndex:
    """Indice de tokens de un documento para contar tokens en un intervalo."""

    __slots__ = ("starts", "ends")

    def __init__(self, offsets: Sequence[tuple[int, int]]):
        self.starts = [int(o[0]) for o in offsets]
        self.ends = [int(o[1]) for o in offsets]
        if any(b < a for a, b in zip(self.starts, self.starts[1:])):
            raise ValueError("Offsets de tokens no monotonos")

    def __len__(self) -> int:
        return len(self.starts)

    def count(self, start: int, end: int) -> int:
        """Tokens cuyo inicio cae en [start, end)."""
        return bisect_left(self.starts, end) - bisect_left(self.starts, start)

    def first_at_or_after(self, pos: int) -> int:
        return bisect_left(self.starts, pos)

    def window_span(self, i: int, j: int) -> tuple[int, int]:
        """Intervalo de caracteres cubierto por los tokens [i, j)."""
        return self.starts[i], self.ends[j - 1]


class HFTokenizerAdapter:
    """Adaptador de un tokenizador rapido de Hugging Face."""

    def __init__(self, tokenizer: Any, name: str):
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Se requiere un tokenizador rapido (offsets de caracteres).")
        self._tok = tokenizer
        self.name = name
        self.num_special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))

    @classmethod
    def from_pretrained(cls, name: str, revision: str | None = None) -> "HFTokenizerAdapter":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
        return cls(tok, name)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return [(int(s), int(e)) for s, e in enc["offset_mapping"] if e > s]

    def count(self, text: str) -> int:
        enc = self._tok(text, add_special_tokens=False, truncation=False, verbose=False)
        return len(enc["input_ids"])


class RegexTokenizerAdapter:
    """Tokenizador aproximado (palabras y signos). SOLO para pruebas y simulacros;
    no representa a ningun modelo real."""

    name = "regex-approx"
    num_special_tokens = 2
    _RE = re.compile(r"\w+|[^\w\s]")

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in self._RE.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in self._RE.finditer(text))
