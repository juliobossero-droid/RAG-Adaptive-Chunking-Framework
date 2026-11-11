import random

import numpy as np
import pytest

from chunklab.config import CommonConfig
from chunklab.core import Document
from chunklab.embedding import HashingEmbedder
from chunklab.sentences import segment_sentences
from chunklab.tokenization import RegexTokenizerAdapter, TokenIndex

TOPICS = {
    "pena": "pena condena prision años cumplimiento vencimiento computo reclusion delito",
    "conducta": "conducta concepto calificacion disciplina buena observada interno penal",
    "psico": "informe psicologico evaluacion pronostico criminologico riesgo personalidad",
    "social": "familia trabajo vivienda arraigo situacion social laboral domicilio apoyo",
    "norma": "articulo ley codigo requisitos normativa aplicable libertad condicional juez",
}


def make_text(seed: int = 0, n_sent: int = 60, with_headings: bool = True) -> str:
    rng = random.Random(seed)
    keys = list(TOPICS)
    sents = []
    topic = rng.choice(keys)
    for i in range(n_sent):
        if rng.random() < 0.15:
            topic = rng.choice(keys)
        words = TOPICS[topic].split()
        body = " ".join(rng.choice(words) for _ in range(rng.randint(10, 22)))
        sents.append(body.capitalize() + ".")
    if not with_headings:
        return " ".join(sents)
    a, b = n_sent // 5, 4 * n_sent // 5
    return (
        "AUTOS Y VISTOS:\n" + " ".join(sents[:a]) + "\n\n"
        "Y CONSIDERANDO:\n" + " ".join(sents[a:b]) + "\n\n"
        "RESUELVO:\n" + " ".join(sents[b:])
    )


def make_doc(doc_id: str = "d1", **kw) -> Document:
    import hashlib

    text = make_text(**kw)
    return Document(doc_id, text, hashlib.sha256(text.encode()).hexdigest())


@pytest.fixture
def common():
    return CommonConfig(min_tokens=20, max_tokens=80, safety_margin_tokens=4)


@pytest.fixture
def tok():
    return RegexTokenizerAdapter()


@pytest.fixture
def emb():
    return HashingEmbedder(dim=128)
