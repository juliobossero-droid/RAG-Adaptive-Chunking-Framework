"""Interfaz de linea de comandos.

  chunklab run --config config.yaml [--limit N] [--force] [--dry-run]
  chunklab export --config config.yaml
  chunklab diagnose-headings --config config.yaml
  chunklab inspect --config config.yaml --doc-id <id> [--strategy <nombre>]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

from .config import ConfigError, load_config
from .export import export_run
from .headings import detect_headings
from .io import load_documents, corpus_fingerprint
from .pipeline import build_runtime, process_document, run_corpus

LOG = logging.getLogger("chunklab")


def _out_dir(cfg) -> Path:
    return Path(cfg.output_dir) / cfg.label


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    docs = load_documents(cfg.corpus)
    if args.limit:
        docs = docs[: args.limit]
    LOG.info("Corpus: %d documentos | huella %s", len(docs), corpus_fingerprint(docs)[:12])
    LOG.info("Configuracion: hash %s", cfg.fingerprint()[:12])
    if args.dry_run:
        print(f"OK: {len(docs)} documentos, estrategias: {[s.name for s in cfg.strategies]}")
        return 0
    rt = build_runtime(cfg)
    out = _out_dir(cfg)
    summary = run_corpus(docs, rt, out, force=args.force, progress=not args.quiet)
    report = export_run(cfg, out, docs, extra={"run_summary": summary, "embedder": rt.embedder.fingerprint(), "hard_limit": rt.hard_limit})
    print(f"Documentos -> procesados: {summary['processed']} | reanudados: {summary['skipped']} | fallidos: {summary['failed']}")
    for row in report["strategies"]:
        print(
            f"  {row['strategy']:<18} chunks={row['n_chunks']:<8} tokens(mediana)={row['tokens_median']:<6} "
            f"p95={row['tokens_p95']:<6} max={row['tokens_max']}"
        )
    print(f"Resultados en: {out}")
    return 1 if summary["failed"] else 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = export_run(cfg, _out_dir(cfg))
    print(f"Exportados {report['n_docs']} documentos; incidencias: {report['n_issues']}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    docs = load_documents(cfg.corpus)
    rows, labels, counts = [], Counter(), Counter()
    for d in docs:
        hs, strict = detect_headings(d.text)
        counts[len(hs)] += 1
        for h in set(x.label for x in hs):
            labels[h] += 1
        rows.append(
            {
                "doc_id": d.doc_id,
                "n_headings": len(hs),
                "strict_mode": strict,
                "sequence": ">".join(h.label for h in hs),
                "offsets": ";".join(f"{h.start}:{h.text}" for h in hs),
            }
        )
    out = _out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "heading_diagnosis.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(docs)
    print(f"Documentos: {n} | sin encabezados: {counts[0]} ({counts[0] / n:.1%})")
    for lbl in ("vistos", "considerando", "resuelvo"):
        print(f"  con '{lbl}': {labels[lbl]} ({labels[lbl] / n:.1%})")
    print(f"Detalle para revision manual: {path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    docs = {d.doc_id: d for d in load_documents(cfg.corpus)}
    if args.doc_id not in docs:
        print(f"doc_id inexistente: {args.doc_id}", file=sys.stderr)
        return 2
    rt = build_runtime(cfg)
    payload = process_document(docs[args.doc_id], rt)
    print("Estadisticas del documento:", payload["doc_stats"])
    for name, chunks in payload["strategies"].items():
        if args.strategy and name != args.strategy:
            continue
        print(f"\n===== {name}: {len(chunks)} chunks =====")
        for c in chunks:
            flags = {k: v for k, v in c["meta"].items() if k in ("section", "fallback", "resuelvo_overflow", "undersized")}
            head = c["text"][:160].replace("\n", " ")
            print(f"[{c['index']:03d}] tok={c['n_tokens']:<4} spans={len(c['spans'])} {flags} | {head}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chunklab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="segmenta el corpus (reanudable)")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--limit", type=int, help="procesa solo los primeros N documentos (prueba rapida)")
    r.add_argument("--force", action="store_true", help="regenera shards existentes")
    r.add_argument("--dry-run", action="store_true", help="valida configuracion y corpus sin procesar")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("export", help="reconstruye chunks/ y estadisticas desde los shards")
    e.add_argument("--config", default="config.yaml")
    e.set_defaults(fn=cmd_export)
    d = sub.add_parser("diagnose-headings", help="diagnostico de deteccion de encabezados (sin modelos)")
    d.add_argument("--config", default="config.yaml")
    d.set_defaults(fn=cmd_diagnose)
    i = sub.add_parser("inspect", help="muestra los chunks de un documento")
    i.add_argument("--config", default="config.yaml")
    i.add_argument("--doc-id", required=True)
    i.add_argument("--strategy")
    i.set_defaults(fn=cmd_inspect)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
