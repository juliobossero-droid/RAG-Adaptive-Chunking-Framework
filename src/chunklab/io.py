"""Lectura canonica del corpus y escritura atomica de archivos."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import unicodedata
from pathlib import Path
from typing import Any

from .config import ConfigError, CorpusConfig
from .core import Document

LOG = logging.getLogger("chunklab")


def canonicalize(raw: str) -> str:
    """Texto canonico: sin BOM, saltos de linea '\\n', Unicode NFC.

    TODOS los offsets (chunks y evidencias anotadas) se refieren a este texto. Las
    herramientas de anotacion deben cargar los documentos con `load_documents`."""
    t = raw.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", t)


def read_text_file(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ConfigError(f"No se pudo decodificar {path}")  # latin-1 nunca falla; defensivo


def load_documents(cfg: CorpusConfig) -> list[Document]:
    root = Path(cfg.documents_dir)
    if not root.is_dir():
        raise ConfigError(f"No existe el directorio de documentos: {root}")
    files = sorted(root.rglob(cfg.pattern))
    by_id = {f.relative_to(root).with_suffix("").as_posix(): f for f in files}
    if not by_id:
        raise ConfigError(f"No hay archivos {cfg.pattern} en {root}")
    ids = sorted(by_id)
    if cfg.document_list:
        wanted = [
            ln.strip()
            for ln in Path(cfg.document_list).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise ConfigError(f"{len(missing)} doc_id de la lista no existen, ej.: {missing[:3]}")
        ids = sorted(set(wanted))
    elif cfg.max_documents is not None and cfg.max_documents < len(ids):
        ids = sorted(random.Random(cfg.sample_seed).sample(ids, cfg.max_documents))
    docs: list[Document] = []
    for doc_id in ids:
        raw, enc = read_text_file(by_id[doc_id])
        text = canonicalize(raw)
        docs.append(
            Document(
                doc_id=doc_id,
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source_path=str(by_id[doc_id]),
                encoding=enc,
            )
        )
    return docs


def corpus_fingerprint(docs: list[Document]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(f"{d.doc_id}:{d.sha256};".encode())
    return h.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def dump_json(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, default=json_default))
