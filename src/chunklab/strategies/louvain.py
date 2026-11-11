"""Segmentacion por grafo de similitud y comunidades de Louvain.

Definicion (fiel a CONAIISI/ReDDI): nodo = oracion; arista entre oraciones con
similitud coseno >= tau (peso = similitud); comunidades por Louvain (resolucion
gamma); cada comunidad es un chunk (puede reunir oraciones NO adyacentes, ordenadas
por posicion en el texto). Cambios respecto de aquellos trabajos, explicitos:
  * comunidades chicas (< min_tokens) se fusionan con la comunidad mas cercana en
    posicion (desempate por similitud de centroides);
  * toda comunidad que exceda max_tokens se divide en piezas equilibradas, para no
    truncar el embedding;
  * `contiguous: true` genera la variante de chunks contiguos (corridas de oraciones
    consecutivas de la misma comunidad); `window: k` restringe aristas a |i-j| <= k.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from ..config import ConfigError
from ..core import RawChunk
from .base import DocView, Strategy, balanced_split, raw_from_indices

MAX_GLOBAL_NODES = 4000


def community_groups(
    emb: np.ndarray, *, tau: float, gamma: float, window: int | None, seed: int
) -> list[list[int]]:
    n = len(emb)
    if window is None and n > MAX_GLOBAL_NODES:
        raise ValueError(
            f"Documento con {n} oraciones: use 'window' para limitar el grafo (>{MAX_GLOBAL_NODES})"
        )
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    sim = emb @ emb.T
    for i in range(n - 1):
        hi = n if window is None else min(n, i + window + 1)
        js = np.nonzero(sim[i, i + 1 : hi] >= tau)[0] + i + 1
        graph.add_weighted_edges_from((i, int(j), float(sim[i, j])) for j in js)
    comms = nx.community.louvain_communities(graph, weight="weight", resolution=gamma, seed=seed)
    groups = [sorted(int(x) for x in c) for c in comms]
    groups.sort(key=lambda g: g[0])
    return groups


def split_into_runs(groups: list[list[int]], n: int) -> list[list[int]]:
    label = np.empty(n, dtype=np.int64)
    for gi, g in enumerate(groups):
        label[g] = gi
    runs: list[list[int]] = []
    cur = [0]
    for i in range(1, n):
        if label[i] == label[i - 1]:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return runs


def _centroid(emb: np.ndarray, g: list[int]) -> np.ndarray:
    c = emb[g].mean(axis=0)
    nrm = float(np.linalg.norm(c))
    return c / nrm if nrm > 0 else c


def merge_small_groups(
    groups: list[list[int]], emb: np.ndarray, tokens: np.ndarray, min_tokens: int
) -> list[list[int]]:
    """Fusiona grupos de menos de min_tokens con el grupo mas cercano en posicion.
    Determinista: se procesa primero el grupo mas chico (luego el mas temprano)."""
    groups = sorted((sorted(g) for g in groups), key=lambda g: g[0])

    def gtok(g: list[int]) -> int:
        return int(sum(int(tokens[i]) for i in g))

    while len(groups) > 1:
        small = [(gtok(g), g[0], k) for k, g in enumerate(groups) if gtok(g) < min_tokens]
        if not small:
            break
        k = min(small)[2]
        g = groups[k]
        arr = np.asarray(g)
        cg = _centroid(emb, g)
        best_key = None
        best_m = -1
        for m, h in enumerate(groups):
            if m == k:
                continue
            d = int(np.abs(arr[:, None] - np.asarray(h)[None, :]).min())
            key = (d, -float(cg @ _centroid(emb, h)), m)
            if best_key is None or key < best_key:
                best_key, best_m = key, m
        merged = sorted(groups[best_m] + g)
        groups = [x for idx, x in enumerate(groups) if idx not in (k, best_m)] + [merged]
        groups.sort(key=lambda x: x[0])
    return groups


class LouvainStrategy(Strategy):
    kind = "louvain_graph"
    needs_embeddings = True
    DEFAULTS = {"tau": 0.72, "gamma": 1.0, "window": None, "contiguous": False, "min_tokens": None}

    def _check(self, p: dict[str, Any]) -> dict[str, Any]:
        out = dict(p)
        out["tau"], out["gamma"] = float(p["tau"]), float(p["gamma"])
        out["window"] = None if p["window"] is None else int(p["window"])
        out["contiguous"] = bool(p["contiguous"])
        out["min_tokens"] = int(p["min_tokens"] or self.common.min_tokens)
        if not -1.0 <= out["tau"] <= 1.0 or out["gamma"] <= 0:
            raise ConfigError(f"'{self.name}': tau en [-1,1] y gamma > 0")
        if out["window"] is not None and out["window"] < 1:
            raise ConfigError(f"'{self.name}': window debe ser >= 1")
        if out["min_tokens"] >= self.common.max_tokens:
            raise ConfigError(f"'{self.name}': min_tokens debe ser < common.max_tokens")
        return out

    def chunk(self, view: DocView) -> list[RawChunk]:
        n = len(view.sentences)
        if n == 0:
            return []
        emb, tokens, p = view.embeddings(), view.sentence_tokens, self.params
        groups = community_groups(emb, tau=p["tau"], gamma=p["gamma"], window=p["window"], seed=self.seed)
        n_comm = len(groups)
        if p["contiguous"]:
            groups = split_into_runs(groups, n)
        groups = merge_small_groups(groups, emb, tokens, p["min_tokens"])
        out: list[RawChunk] = []
        for g in groups:
            pieces = balanced_split(g, tokens, self.common.max_tokens)
            for piece in pieces:
                out.append(
                    raw_from_indices(
                        view,
                        piece,
                        {"group_sentences": len(g), "split_pieces": len(pieces), "n_communities": n_comm},
                    )
                )
        return out
