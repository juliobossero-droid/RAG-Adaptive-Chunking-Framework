import json
from pathlib import Path

import pytest
import yaml

from conftest import make_text
from chunklab.cli import main
from chunklab.config import ConfigError, load_config
from chunklab.io import canonicalize, load_documents
from chunklab.pipeline import build_runtime, run_corpus


def write_corpus(root: Path, n=6):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"exp_{i:02d}.txt").write_text(make_text(seed=i, n_sent=50), encoding="utf-8")
    (root / "sin_enc.txt").write_text(make_text(seed=99, n_sent=40, with_headings=False), encoding="utf-8")


def write_config(tmp: Path, **over):
    cfg = {
        "run": {"label": "t", "seed": 42, "output_dir": str(tmp / "runs")},
        "corpus": {"documents_dir": str(tmp / "docs"), "pattern": "*.txt"},
        "embedding": {"backend": "hashing"},
        "common": {"min_tokens": 20, "max_tokens": 80, "safety_margin_tokens": 4},
        "strategies": {
            "fixed": {"kind": "fixed_tokens", "window_tokens": 60, "overlap_tokens": 12},
            "fixed_nooverlap": {"kind": "fixed_tokens", "window_tokens": 60, "overlap_tokens": 0},
            "sentence_pack": {"kind": "sentence_pack", "target_tokens": 60},
            "semantic": {"kind": "semantic_dp"},
            "louvain": {"kind": "louvain_graph", "tau": 0.3},
            "legal": {"kind": "legal_structural"},
        },
    }
    cfg.update(over)
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_end_to_end_resume_and_export(tmp_path):
    write_corpus(tmp_path / "docs")
    cfgp = write_config(tmp_path)
    assert main(["run", "--config", str(cfgp), "--quiet"]) == 0
    out = tmp_path / "runs" / "t"
    assert len(list((out / "shards").glob("*.json"))) == 7
    for name in ("fixed", "fixed_nooverlap", "sentence_pack", "semantic", "louvain", "legal"):
        lines = (out / "chunks" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert lines and json.loads(lines[0])["strategy"] == name
    assert (out / "chunk_stats.csv").exists() and (out / "manifest.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_summary"]["failed"] == 0
    # segunda corrida: todo se reanuda (nada se reprocesa)
    cfg = load_config(cfgp)
    docs = load_documents(cfg.corpus)
    rt = build_runtime(cfg)
    s = run_corpus(docs, rt, out, progress=False)
    assert s == {"processed": 0, "skipped": 7, "failed": 0}
    # el documento sin encabezados cae en fallback
    stats = (out / "doc_stats.csv").read_text(encoding="utf-8")
    assert "True" in stats


def test_resume_after_interruption(tmp_path):
    write_corpus(tmp_path / "docs")
    cfgp = write_config(tmp_path)
    cfg = load_config(cfgp)
    docs = load_documents(cfg.corpus)
    rt = build_runtime(cfg)
    out = tmp_path / "runs" / "t"
    run_corpus(docs[:3], rt, out, progress=False)
    s = run_corpus(docs, rt, out, progress=False)
    assert s["skipped"] == 3 and s["processed"] == 4


def test_config_change_blocks_reuse(tmp_path):
    write_corpus(tmp_path / "docs", n=2)
    cfgp = write_config(tmp_path)
    assert main(["run", "--config", str(cfgp), "--quiet"]) == 0
    write_config(tmp_path, common={"min_tokens": 20, "max_tokens": 70, "safety_margin_tokens": 4})
    with pytest.raises(ConfigError):
        cfg = load_config(cfgp)
        run_corpus(load_documents(cfg.corpus), build_runtime(cfg), tmp_path / "runs" / "t", progress=False)


def test_token_limit_guard(tmp_path):
    write_corpus(tmp_path / "docs", n=1)
    cfgp = write_config(tmp_path)
    cfg = load_config(cfgp)
    from chunklab.embedding import HashingEmbedder

    import chunklab.pipeline as pl

    orig = pl.HashingEmbedder
    pl.HashingEmbedder = lambda: HashingEmbedder(effective_token_limit=50)
    try:
        with pytest.raises(ConfigError, match="truncan"):
            build_runtime(cfg)
    finally:
        pl.HashingEmbedder = orig


def test_canonicalize_and_sampling_reproducible(tmp_path):
    assert canonicalize("\ufeffa\r\nb\rc") == "a\nb\nc"
    write_corpus(tmp_path / "docs", n=10)
    cfgp = write_config(tmp_path, corpus={"documents_dir": str(tmp_path / "docs"), "max_documents": 4, "sample_seed": 7})
    a = [d.doc_id for d in load_documents(load_config(cfgp).corpus)]
    b = [d.doc_id for d in load_documents(load_config(cfgp).corpus)]
    assert a == b and len(a) == 4


def test_unknown_config_keys_rejected(tmp_path):
    write_corpus(tmp_path / "docs", n=1)
    cfgp = write_config(tmp_path, embedding={"backend": "hashing", "modelo": "x"})
    with pytest.raises(ConfigError):
        load_config(cfgp)
