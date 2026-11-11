"""Pipeline de chunking: por documento, reanudable y con escritura atomica."""

from __future__ import annotations

import json
import logging
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import Config, ConfigError
from .core import ChunkingError, Document, doc_key
from .embedding import Embedder, EmbeddingCache, HashingEmbedder, SentenceTransformerEmbedder
from .headings import detect_headings
from .io import atomic_write_text, dump_json, json_default
from .sentences import DEFAULT_ABBREVIATIONS, segment_sentences, split_oversized
from .strategies import DocView, Strategy, build_strategies, finalize
from .tokenization import HFTokenizerAdapter, RegexTokenizerAdapter, TokenIndex, TokenizerAdapter
from .validate import validate_chunks

LOG = logging.getLogger("chunklab")
SCHEMA = 1


@dataclass
class Runtime:
    cfg: Config
    tokenizer: TokenizerAdapter
    embedder: Embedder
    strategies: list[Strategy]
    abbreviations: frozenset[str]
    cache: EmbeddingCache
    hard_limit: int


def build_runtime(cfg: Config, *, need_models: bool = True) -> Runtime:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:  # semilla de torch si esta disponible
        import torch

        torch.manual_seed(cfg.seed)
    except ImportError:
        pass
    emb_cfg = cfg.embedding
    if emb_cfg.backend == "hashing":
        LOG.warning("BACKEND 'hashing': solo para simulacros; los resultados NO tienen validez cientifica")
        embedder: Embedder = HashingEmbedder()
        tokenizer: TokenizerAdapter = RegexTokenizerAdapter()
    else:
        st = SentenceTransformerEmbedder(
            emb_cfg.model,
            device=emb_cfg.device,
            batch_size=emb_cfg.batch_size,
            prefix=emb_cfg.sentence_prefix,
            revision=emb_cfg.revision,
        )
        embedder = st
        tokenizer = HFTokenizerAdapter(st.tokenizer, emb_cfg.model)
        LOG.info(
            "Modelo %s | max_seq_length=%s | limite efectivo=%s tokens",
            emb_cfg.model, st.max_seq_length, st.effective_token_limit,
        )
    hard_limit = embedder.effective_token_limit
    c = cfg.common
    if c.max_tokens + c.safety_margin_tokens > hard_limit:
        raise ConfigError(
            f"max_tokens ({c.max_tokens}) + margen ({c.safety_margin_tokens}) excede el limite efectivo "
            f"del modelo ({hard_limit}). Baje max_tokens o use un modelo de contexto mas largo: "
            f"de lo contrario los chunks se truncan en el embedding."
        )
    return Runtime(
        cfg=cfg,
        tokenizer=tokenizer,
        embedder=embedder,
        strategies=build_strategies(cfg.strategies, cfg.common, cfg.seed),
        abbreviations=DEFAULT_ABBREVIATIONS | frozenset(cfg.extra_abbreviations),
        cache=EmbeddingCache(Path(cfg.cache_dir) / "embeddings", embedder),
        hard_limit=hard_limit,
    )


def process_document(doc: Document, rt: Runtime) -> dict[str, Any]:
    """Segmenta un documento con todas las estrategias habilitadas y lo valida."""
    text = doc.text
    if not text.strip():
        raise ChunkingError("documento vacio")
    ti = TokenIndex(rt.tokenizer.offsets(text))
    headings, strict = detect_headings(text)
    raw_sents = segment_sentences(
        text, abbreviations=rt.abbreviations, forced_boundaries=[h.start for h in headings]
    )
    sents, atom = split_oversized(raw_sents, ti, text, rt.cfg.common.max_tokens)
    sent_tokens = np.array([ti.count(s.start, s.end) for s in sents], dtype=np.int64)

    def embed() -> np.ndarray:
        texts = [" ".join(text[s.start : s.end].split()) for s in sents]
        return rt.cache.get(doc, sents, texts)

    view = DocView(doc, ti, sents, sent_tokens, headings, strict, _embed_fn=embed)
    payload_chunks: dict[str, list[dict]] = {}
    issues: list[dict] = []
    fallback = False
    for st in rt.strategies:
        chunks = finalize(view, st.chunk(view), st.name)
        found = validate_chunks(
            doc, chunks, strategy=st.name, max_tokens=rt.cfg.common.max_tokens,
            hard_limit=rt.hard_limit, tokenizer=rt.tokenizer,
            coverage_mode=st.coverage_mode, allows_overlap=st.allows_overlap,
        )
        errors = [i for i in found if i.severity == "error"]
        if errors:
            raise ChunkingError(f"{st.name}: " + "; ".join(f"{i.code}: {i.message}" for i in errors[:5]))
        issues.extend(i.to_dict() for i in found)
        payload_chunks[st.name] = [c.to_dict() for c in chunks]
        if st.kind == "legal_structural":
            fallback = any(c.meta.get("fallback") for c in chunks)
    stats = {
        "chars": len(text),
        "n_tokens": len(ti),
        "n_sentences": len(sents),
        "sentence_tokens_mean": float(sent_tokens.mean()) if len(sent_tokens) else 0.0,
        "sentence_tokens_max": int(sent_tokens.max()) if len(sent_tokens) else 0,
        "oversized_sentences": atom["oversized"],
        "atomized_soft_cuts": atom["soft"],
        "atomized_hard_cuts": atom["hard"],
        "n_headings": len(headings),
        "heading_labels": [h.label for h in headings],
        "strict_headings": strict,
        "legal_fallback": fallback,
        "encoding": doc.encoding,
    }
    return {
        "schema": SCHEMA,
        "doc_id": doc.doc_id,
        "doc_sha256": doc.sha256,
        "config_hash": rt.cfg.fingerprint(),
        "doc_stats": stats,
        "issues": issues,
        "strategies": payload_chunks,
    }


def shard_path(out_dir: Path, doc_id: str) -> Path:
    return Path(out_dir) / "shards" / f"{doc_key(doc_id)}.json"


def run_corpus(
    docs: list[Document], rt: Runtime, out_dir: Path, *, force: bool = False, progress: bool = True
) -> dict[str, int]:
    """Procesa el corpus. Reanuda: omite documentos con shard valido (mismo hash de
    configuracion y de documento). Si el shard existe con otra configuracion, aborta."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = rt.cfg.fingerprint()
    done = skipped = failed = 0
    errors_path = out_dir / "errors.jsonl"
    for doc in tqdm(docs, desc="Documentos", disable=not progress):
        sp = shard_path(out_dir, doc.doc_id)
        if sp.exists() and not force:
            try:
                head = json.loads(sp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                head = {}
            if head.get("config_hash") == fp and head.get("doc_sha256") == doc.sha256:
                skipped += 1
                continue
            raise ConfigError(
                f"{sp.name} fue generado con otra configuracion o version del documento. "
                f"Use un 'run.label' nuevo o --force para regenerar."
            )
        try:
            atomic_write_text(sp, json.dumps(process_document(doc, rt), ensure_ascii=False, default=json_default))
            done += 1
        except Exception as exc:  # se registra y se continua; el exit code final es distinto de 0
            failed += 1
            LOG.error("Fallo en %s: %s", doc.doc_id, exc)
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"doc_id": doc.doc_id, "error": str(exc), "trace": traceback.format_exc()}, ensure_ascii=False) + "\n")
    return {"processed": done, "skipped": skipped, "failed": failed}
