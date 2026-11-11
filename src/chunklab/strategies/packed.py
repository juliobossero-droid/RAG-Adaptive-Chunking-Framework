"""Control: empaquetado voraz de oraciones SIN decision semantica."""

from __future__ import annotations

from typing import Any

from ..config import ConfigError
from ..core import RawChunk
from .base import DocView, Strategy, raw_from_indices


class SentencePackStrategy(Strategy):
    """Agrupa oraciones consecutivas hasta `target_tokens`. Sirve para saber si las
    estrategias semanticas aportan algo mas alla de respetar limites de oracion."""

    kind = "sentence_pack"
    DEFAULTS = {"target_tokens": None}

    def _check(self, p: dict[str, Any]) -> dict[str, Any]:
        t = int(p["target_tokens"] or self.common.max_tokens)
        if not self.common.min_tokens <= t <= self.common.max_tokens:
            raise ConfigError(f"'{self.name}': target_tokens debe estar en [min_tokens, max_tokens]")
        return {"target_tokens": t}

    def chunk(self, view: DocView) -> list[RawChunk]:
        target, cap = self.params["target_tokens"], self.common.max_tokens
        tokens = view.sentence_tokens
        groups: list[list[int]] = []
        cur: list[int] = []
        cur_t = 0
        for i in range(len(view.sentences)):
            t = int(tokens[i])
            if cur and cur_t + t > target:
                groups.append(cur)
                cur, cur_t = [], 0
            cur.append(i)
            cur_t += t
        if cur:
            groups.append(cur)
        # cola pequena: se fusiona con el grupo anterior si no supera el tope
        if len(groups) > 1 and sum(int(tokens[i]) for i in groups[-1]) < self.common.min_tokens:
            merged = groups[-2] + groups[-1]
            if sum(int(tokens[i]) for i in merged) <= cap:
                groups[-2:] = [merged]
        return [raw_from_indices(view, g) for g in groups]
