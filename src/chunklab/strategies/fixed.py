
"""Ventanas fijas de tokens (con o sin solapamiento)."""

from __future__ import annotations

from typing import Any

from ..config import ConfigError
from ..core import RawChunk, Span
from .base import DocView, Strategy


class FixedTokenStrategy(Strategy):
    """Ventanas fijas de tokens, con o sin solapamiento."""

    kind = "fixed_tokens"
    coverage_mode = "tokens"
    DEFAULTS = {"window_tokens": None, "overlap_tokens": 50}

    def _check(self, p: dict[str, Any]) -> dict[str, Any]:
        w = int(p["window_tokens"] or self.common.max_tokens)
        o = int(p["overlap_tokens"])

        if w > self.common.max_tokens:
            raise ConfigError(
                f"'{self.name}': window_tokens ({w}) > common.max_tokens"
            )

        if not 0 <= o < w:
            raise ConfigError(
                f"'{self.name}': se requiere 0 <= overlap_tokens < window_tokens"
            )

        return {"window_tokens": w, "overlap_tokens": o}

    @property
    def allows_overlap(self) -> bool:
        return self.params["overlap_tokens"] > 0

    def chunk(self, view: DocView) -> list[RawChunk]:
        ti = view.token_index
        n = len(ti)

        w, o = self.params["window_tokens"], self.params["overlap_tokens"]

        # Margen preventivo frente a diferencias de retokenización.
        effective_w = min(
            w,
            self.common.max_tokens - self.common.safety_margin_tokens,
        )

        if effective_w <= o:
            raise ConfigError(
                f"'{self.name}': el solapamiento debe ser menor "
                f"que la ventana efectiva ({effective_w})"
            )

        w = effective_w
        stride = w - o

        out: list[RawChunk] = []
        i = 0

        while n > 0:
            j = min(i + w, n)
            s, e = ti.window_span(i, j)

            out.append(
                RawChunk(
                    (Span(s, e),),
                    0,
                    {"token_start": i, "token_end": j},
                )
            )

            if j == n:
                break

            i += stride

        return out
