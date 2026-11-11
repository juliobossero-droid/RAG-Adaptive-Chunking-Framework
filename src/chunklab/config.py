"""Carga y validacion de la configuracion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .headings import HEADINGS_VERSION
from .sentences import SEGMENTER_VERSION


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusConfig:
    documents_dir: str
    pattern: str = "*.txt"
    max_documents: int | None = None
    sample_seed: int = 42
    document_list: str | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "sentence-transformers"  # | "hashing" (solo simulacros)
    model: str = "intfloat/multilingual-e5-base"
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = 32
    sentence_prefix: str = "query: "


@dataclass(frozen=True)
class CommonConfig:
    """Rango de tamano comun a TODAS las estrategias (medido en tokens del modelo)."""

    min_tokens: int = 64
    max_tokens: int = 256
    safety_margin_tokens: int = 8


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    label: str
    seed: int
    output_dir: str
    cache_dir: str
    corpus: CorpusConfig
    embedding: EmbeddingConfig
    common: CommonConfig
    extra_abbreviations: tuple[str, ...]
    strategies: tuple[StrategySpec, ...]
    raw: dict[str, Any]

    def fingerprint(self) -> str:
        """Hash de todo lo que afecta el RESULTADO del chunking (no rutas ni dispositivo)."""
        payload = {
            "seed": self.seed,
            "embedding": {
                "backend": self.embedding.backend,
                "model": self.embedding.model,
                "revision": self.embedding.revision,
                "prefix": self.embedding.sentence_prefix,
            },
            "common": vars(self.common),
            "abbr": sorted(self.extra_abbreviations),
            "strategies": [[s.name, s.kind, s.params] for s in self.strategies],
            "segmenter": SEGMENTER_VERSION,
            "headings": HEADINGS_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


def _need(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Falta '{key}' en la seccion '{ctx}'")
    return d[key]


def _path(base: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path: str | Path) -> Config:
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("El YAML debe ser un mapeo")
    base = p.parent
    run = _need(raw, "run", "raiz")
    corpus = _need(raw, "corpus", "raiz")
    out = _path(base, run.get("output_dir", "./runs"))
    cfg_corpus = CorpusConfig(
        documents_dir=_path(base, _need(corpus, "documents_dir", "corpus")),
        pattern=corpus.get("pattern", "*.txt"),
        max_documents=corpus.get("max_documents"),
        sample_seed=int(corpus.get("sample_seed", 42)),
        document_list=_path(base, corpus["document_list"]) if corpus.get("document_list") else None,
    )
    emb = raw.get("embedding", {})
    cfg_emb = EmbeddingConfig(**{k: v for k, v in emb.items() if k in EmbeddingConfig.__dataclass_fields__})
    unknown = set(emb) - set(EmbeddingConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"Claves desconocidas en 'embedding': {sorted(unknown)}")
    if cfg_emb.backend not in ("sentence-transformers", "hashing"):
        raise ConfigError(f"backend no soportado: {cfg_emb.backend}")
    com = raw.get("common", {})
    unknown = set(com) - set(CommonConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"Claves desconocidas en 'common': {sorted(unknown)}")
    cfg_com = CommonConfig(**com)
    if not (0 < cfg_com.min_tokens < cfg_com.max_tokens):
        raise ConfigError("Se requiere 0 < min_tokens < max_tokens")
    specs = []
    for name, body in _need(raw, "strategies", "raiz").items():
        body = dict(body or {})
        if not body.pop("enabled", True):
            continue
        kind = body.pop("kind", None)
        if kind is None:
            raise ConfigError(f"La estrategia '{name}' no declara 'kind'")
        specs.append(StrategySpec(name=name, kind=kind, params=body))
    if not specs:
        raise ConfigError("No hay estrategias habilitadas")
    return Config(
        label=str(run.get("label", "run")),
        seed=int(run.get("seed", 42)),
        output_dir=out,
        cache_dir=_path(base, run["cache_dir"]) if run.get("cache_dir") else str(Path(out) / "_cache"),
        corpus=cfg_corpus,
        embedding=cfg_emb,
        common=cfg_com,
        extra_abbreviations=tuple(a.lower().rstrip(".") for a in raw.get("sentences", {}).get("extra_abbreviations", [])),
        strategies=tuple(specs),
        raw=raw,
    )
