"""Consolidacion de shards: chunks por estrategia, estadisticas y manifiesto."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import Config
from .io import atomic_write_text, corpus_fingerprint, dump_json
from .core import Document


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_run(cfg: Config, out_dir: Path, docs: list[Document] | None = None, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    shards = sorted((out_dir / "shards").glob("*.json"))
    per_strategy: dict[str, list[dict]] = {}
    doc_rows: list[dict] = []
    issue_rows: list[dict] = []
    for sp in shards:
        d = json.loads(sp.read_text(encoding="utf-8"))
        if d.get("config_hash") != cfg.fingerprint():
            raise ValueError(f"{sp.name}: hash de configuracion distinto; no se puede exportar")
        st = d["doc_stats"]
        doc_rows.append({"doc_id": d["doc_id"], **{k: (";".join(v) if isinstance(v, list) else v) for k, v in st.items()}})
        issue_rows.extend(d["issues"])
        for name, chunks in d["strategies"].items():
            per_strategy.setdefault(name, []).extend(chunks)
    doc_rows.sort(key=lambda r: r["doc_id"])
    stat_rows = []
    for name, chunks in per_strategy.items():
        chunks.sort(key=lambda c: (c["doc_id"], c["index"]))
        lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n"
        atomic_write_text(out_dir / "chunks" / f"{name}.jsonl", lines)
        tok = np.array([c["n_tokens"] for c in chunks], dtype=float)
        n_docs = len({c["doc_id"] for c in chunks})
        meta = [c["meta"] for c in chunks]
        stat_rows.append(
            {
                "strategy": name,
                "n_docs": n_docs,
                "n_chunks": len(chunks),
                "chunks_per_doc": round(len(chunks) / n_docs, 3),
                "tokens_mean": round(float(tok.mean()), 2),
                "tokens_median": float(np.median(tok)),
                "tokens_p05": float(np.percentile(tok, 5)),
                "tokens_p95": float(np.percentile(tok, 95)),
                "tokens_max": int(tok.max()),
                "share_below_min_tokens": round(float((tok < cfg.common.min_tokens).mean()), 4),
                "n_noncontiguous": sum(1 for c in chunks if len(c["spans"]) > 1),
                "n_fallback": sum(1 for m in meta if m.get("fallback")),
                "n_resuelvo_overflow": sum(1 for m in meta if m.get("resuelvo_overflow")),
                "n_undersized": sum(1 for m in meta if m.get("undersized")),
            }
        )
    _write_csv(out_dir / "chunk_stats.csv", stat_rows)
    _write_csv(out_dir / "doc_stats.csv", doc_rows)
    _write_csv(out_dir / "issues.csv", issue_rows)
    libs = {}
    for name in ("numpy", "PyYAML", "networkx", "transformers", "sentence-transformers", "torch", "tqdm"):
        try:
            libs[name] = version(name)
        except PackageNotFoundError:
            libs[name] = None
    manifest = {
        "chunklab": __version__,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libs,
        "config_hash": cfg.fingerprint(),
        "config": cfg.raw,
        "n_docs_exported": len(doc_rows),
        "corpus_fingerprint": corpus_fingerprint(docs) if docs else None,
        "strategies": sorted(per_strategy),
        **(extra or {}),
    }
    dump_json(out_dir / "manifest.json", manifest)
    return {"strategies": stat_rows, "n_docs": len(doc_rows), "n_issues": len(issue_rows)}
