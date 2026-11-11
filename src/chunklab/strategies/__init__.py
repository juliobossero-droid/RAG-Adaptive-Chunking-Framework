"""Registro de estrategias."""

from __future__ import annotations

from ..config import CommonConfig, ConfigError, StrategySpec
from .base import DocView, Strategy, finalize
from .fixed import FixedTokenStrategy
from .legal import LegalStructuralStrategy
from .louvain import LouvainStrategy
from .packed import SentencePackStrategy
from .semantic import SemanticDPStrategy

KINDS: dict[str, type[Strategy]] = {
    cls.kind: cls
    for cls in (
        FixedTokenStrategy,
        SentencePackStrategy,
        SemanticDPStrategy,
        LouvainStrategy,
        LegalStructuralStrategy,
    )
}


def build_strategies(specs: tuple[StrategySpec, ...], common: CommonConfig, seed: int) -> list[Strategy]:
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ConfigError("Nombres de estrategia duplicados")
    out = []
    for s in specs:
        if s.kind not in KINDS:
            raise ConfigError(f"kind desconocido '{s.kind}' en '{s.name}'. Disponibles: {sorted(KINDS)}")
        out.append(KINDS[s.kind](s.name, dict(s.params), common, seed))
    return out


__all__ = ["DocView", "Strategy", "finalize", "build_strategies", "KINDS"]
