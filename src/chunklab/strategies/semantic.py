"""Heuristica semantica con optimizacion EXACTA por programacion dinamica.

Objetivo (por chunk C de n oraciones):
    score(C) = Coh(C) - lam * Pen(C)
    Coh(C)   = similitud coseno media entre todos los pares de oraciones de C
    Pen(C)   = |n - ideal| / (max_sentences - min_sentences)

Como el score de una particion es aditivo por chunk, el optimo global se obtiene en
O(N * max_sentences) (no hace falta una heuristica voraz). Dos aclaraciones:

 1. Objetivo 'sentence_weighted' (por defecto): cada chunk aporta n * score(C). Si se
    sumaran scores sin ponderar, mientras Coh - lam*Pen sea positivo el optimo
    preferiria SIEMPRE mas chunks (cada chunk suma un termino positivo). La
    ponderacion por n elimina ese sesgo, porque el total de oraciones es fijo.
    'sum' reproduce la lectura literal (con sesgo) y queda disponible para analisis
    de sensibilidad.
 2. La forma exacta de Pen no esta especificada en los articulos previos; la usada
    aqui es una normalizacion lineal declarada y debe tratarse como un parametro mas.

Restricciones duras por chunk: tokens <= max_tokens. El minimo de oraciones es
blando (costo `undersize_cost` por chunk subdimensionado): se incumple solo cuando es
inevitable (documento muy corto u oraciones muy largas).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ConfigError
from ..core import RawChunk
from .base import DocView, Strategy, raw_from_indices

_OBJECTIVES = ("sentence_weighted", "sum")


def partition_dp(
    emb: np.ndarray,
    tokens: np.ndarray,
    *,
    max_tokens: int,
    min_sentences: int,
    max_sentences: int,
    ideal_sentences: int,
    lam: float,
    objective: str = "sentence_weighted",
    undersize_cost: float = 1000.0,
) -> tuple[list[tuple[int, int]], float]:
    """Particion optima de N oraciones. Devuelve ([(i, j), ...] semiabiertos, valor)."""
    n_sent = len(tokens)
    if n_sent == 0:
        return [], 0.0
    if emb.shape[0] != n_sent:
        raise ValueError("emb y tokens deben tener la misma cantidad de filas")
    prefix_e = np.zeros((n_sent + 1, emb.shape[1]), dtype=np.float64)
    prefix_e[1:] = np.cumsum(emb, axis=0)
    prefix_t = np.zeros(n_sent + 1, dtype=np.int64)
    prefix_t[1:] = np.cumsum(tokens)
    span = max(max_sentences - min_sentences, 1)
    neg = -np.inf
    best = np.full(n_sent + 1, neg)
    best[0] = 0.0
    back = np.zeros(n_sent + 1, dtype=np.int64)
    for j in range(1, n_sent + 1):
        idx = np.arange(max(0, j - max_sentences), j)
        n = (j - idx).astype(np.float64)
        ok = ((prefix_t[j] - prefix_t[idx]) <= max_tokens) | (n == 1)
        sums = prefix_e[j] - prefix_e[idx]
        sq = np.einsum("ij,ij->i", sums, sums)
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.where(n > 1, (sq - n) / (n * (n - 1)), 0.0)
        quality = coh - lam * (np.abs(n - ideal_sentences) / span)
        seg = quality * n if objective == "sentence_weighted" else quality
        seg = seg - undersize_cost * (n < min_sentences)
        cand = np.where(ok, best[idx] + seg, neg)
        k = int(np.argmax(cand))  # empate -> primer indice -> chunk mas largo
        best[j] = cand[k]
        back[j] = idx[k]
    parts: list[tuple[int, int]] = []
    j = n_sent
    while j > 0:
        i = int(back[j])
        parts.append((i, j))
        j = i
    parts.reverse()
    return parts, float(best[n_sent])


class SemanticDPStrategy(Strategy):
    kind = "semantic_dp"
    needs_embeddings = True
    DEFAULTS = {
        "min_sentences": 3,
        "max_sentences": 15,
        "ideal_sentences": 9,
        "lam": 0.4,
        "objective": "sentence_weighted",
        "undersize_cost": 1000.0,
    }

    def _check(self, p: dict[str, Any]) -> dict[str, Any]:
        return check_dp_params(self.name, p)

    def chunk(self, view: DocView) -> list[RawChunk]:
        parts, _ = partition_dp(
            view.embeddings(), view.sentence_tokens, max_tokens=self.common.max_tokens, **self.params
        )
        ms = self.params["min_sentences"]
        return [raw_from_indices(view, range(i, j), {"undersized": (j - i) < ms}) for i, j in parts]


def check_dp_params(name: str, p: dict[str, Any]) -> dict[str, Any]:
    out = dict(p)
    for k in ("min_sentences", "max_sentences", "ideal_sentences"):
        out[k] = int(p[k])
    out["lam"] = float(p["lam"])
    out["undersize_cost"] = float(p["undersize_cost"])
    if not 1 <= out["min_sentences"] <= out["ideal_sentences"] <= out["max_sentences"]:
        raise ConfigError(f"'{name}': se requiere 1 <= min <= ideal <= max sentencias")
    if out["lam"] < 0:
        raise ConfigError(f"'{name}': lam debe ser >= 0")
    if out["objective"] not in _OBJECTIVES:
        raise ConfigError(f"'{name}': objective debe ser uno de {_OBJECTIVES}")
    return out
