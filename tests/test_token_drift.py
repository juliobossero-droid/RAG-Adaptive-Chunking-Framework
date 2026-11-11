
"""Prueba de regresión de TOKEN_DRIFT con el modelo real."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from chunklab.config import CommonConfig
from chunklab.core import Document
from chunklab.io import read_text_file, canonicalize
from chunklab.strategies import DocView, KINDS, finalize
from chunklab.tokenization import HFTokenizerAdapter, TokenIndex


@pytest.mark.parametrize("overlap", [0, 50])
def test_fixed_real_token_limit(overlap):
    """La estrategia fija debe respetar 256 tokens reales."""

    path = Path("C:/LC/Doc 0000000003.txt")
    assert path.exists(), "No se encontró el documento original"

    # Leer y normalizar el documento.
    raw, encoding = read_text_file(path)
    text = canonicalize(raw)

    doc = Document(
        doc_id=path.stem,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_path=str(path),
        encoding=encoding,
    )

    # Cargar el tokenizador real.
    tokenizer = HFTokenizerAdapter.from_pretrained(
        "intfloat/multilingual-e5-base"
    )

    token_index = TokenIndex(tokenizer.offsets(doc.text))

    common = CommonConfig(
        min_tokens=64,
        max_tokens=256,
        safety_margin_tokens=8,
    )

    # Preparar la vista del documento.
    view = DocView(
        doc=doc,
        token_index=token_index,
        sentences=[],
        sentence_tokens=np.array([], dtype=int),
        headings=[],
        strict_headings=False,
    )

    # Ejecutar la estrategia fija.
    strategy = KINDS["fixed_tokens"](
        "fixed",
        {"window_tokens": 256, "overlap_tokens": overlap},
        common,
        seed=42,
    )

    chunks = finalize(view, strategy.chunk(view), "fixed")

    # Verificar los tokens reales de cada fragmento.
    oversized = [
        (c.index, tokenizer.count(c.text))
        for c in chunks
        if tokenizer.count(c.text) > 256
    ]

    assert not oversized, (
        f"TOKEN_DRIFT con overlap={overlap}: {oversized}"
    )
