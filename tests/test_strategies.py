import numpy as np
import pytest

from conftest import make_doc
from chunklab.config import ConfigError
from chunklab.headings import build_sections, detect_headings
from chunklab.sentences import segment_sentences, split_oversized
from chunklab.strategies import DocView, KINDS, finalize
from chunklab.tokenization import TokenIndex
from chunklab.validate import validate_chunks


def make_view(doc, tok, emb, common):
    ti = TokenIndex(tok.offsets(doc.text))
    heads, strict = detect_headings(doc.text)
    sents = segment_sentences(doc.text, forced_boundaries=[h.start for h in heads])
    sents, _ = split_oversized(sents, ti, doc.text, common.max_tokens)
    st = np.array([ti.count(s.start, s.end) for s in sents])
    texts = [" ".join(doc.text[s.start : s.end].split()) for s in sents]
    return DocView(doc, ti, sents, st, heads, strict, _embed_fn=lambda: emb.encode(texts))


SPECS = {
    "fixed": ("fixed_tokens", {"window_tokens": 60, "overlap_tokens": 12}),
    "fixed_no": ("fixed_tokens", {"window_tokens": 60, "overlap_tokens": 0}),
    "pack": ("sentence_pack", {"target_tokens": 60}),
    "semantic": ("semantic_dp", {}),
    "louvain": ("louvain_graph", {"tau": 0.3}),
    "louvain_c": ("louvain_graph", {"tau": 0.3, "contiguous": True}),
    "legal": ("legal_structural", {}),
}


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invariants_for_every_strategy(name, seed, tok, emb, common):
    doc = make_doc(f"d{seed}", seed=seed, n_sent=70)
    view = make_view(doc, tok, emb, common)
    kind, params = SPECS[name]
    st = KINDS[kind](name, params, common, seed=42)
    chunks = finalize(view, st.chunk(view), name)
    assert chunks
    issues = validate_chunks(
        doc, chunks, strategy=name, max_tokens=common.max_tokens, hard_limit=10_000,
        tokenizer=tok, coverage_mode=st.coverage_mode, allows_overlap=st.allows_overlap,
    )
    assert [i for i in issues if i.severity == "error"] == []
    assert all(c.n_tokens <= common.max_tokens for c in chunks)


def test_fixed_windows_and_no_redundant_tail(tok, emb, common):
    doc = make_doc(seed=3, n_sent=40)
    view = make_view(doc, tok, emb, common)
    st = KINDS["fixed_tokens"]("f", {"window_tokens": 50, "overlap_tokens": 10}, common, 1)
    ch = finalize(view, st.chunk(view), "f")
    n = len(view.token_index)
    assert ch[0].n_tokens == 50
    assert ch[-1].end == view.token_index.ends[-1]
    # ningun chunk esta totalmente contenido en el anterior
    assert all(b.end > a.end for a, b in zip(ch, ch[1:]))
    # solapamiento declarado entre consecutivos
    assert all(a.end > b.start for a, b in zip(ch, ch[1:]))
    assert len(ch) == 1 + -(-(n - 50) // 40)


def test_louvain_is_deterministic_and_noncontiguous_allowed(tok, emb, common):
    doc = make_doc(seed=5, n_sent=80)
    view = make_view(doc, tok, emb, common)
    mk = lambda: KINDS["louvain_graph"]("l", {"tau": 0.3}, common, 7).chunk(view)
    a, b = mk(), mk()
    assert [r.spans for r in a] == [r.spans for r in b]
    cont = KINDS["louvain_graph"]("lc", {"tau": 0.3, "contiguous": True}, common, 7).chunk(view)
    assert all(len(r.spans) == 1 for r in cont)


def test_legal_sections_whole_when_they_fit(tok, emb, common):
    doc = make_doc(seed=2, n_sent=12)
    view = make_view(doc, tok, emb, common)
    st = KINDS["legal_structural"]("legal", {}, common, 1)
    ch = finalize(view, st.chunk(view), "legal")
    labels = [c.meta["section"] for c in ch]
    assert labels[0] == "vistos" and "considerando" in labels and labels[-1] == "resuelvo"
    whole = [c for c in ch if c.meta.get("section_whole")]
    assert whole and all(c.n_tokens <= common.max_tokens for c in whole)


def test_legal_marks_overflow_and_fallback(tok, emb, common):
    doc = make_doc(seed=4, n_sent=200)
    view = make_view(doc, tok, emb, common)
    ch = finalize(view, KINDS["legal_structural"]("legal", {}, common, 1).chunk(view), "legal")
    assert any(c.meta.get("resuelvo_overflow") for c in ch)
    plain = make_doc("plain", seed=1, n_sent=40, with_headings=False)
    v2 = make_view(plain, tok, emb, common)
    ch2 = finalize(v2, KINDS["legal_structural"]("legal", {}, common, 1).chunk(v2), "legal")
    assert ch2 and all(c.meta.get("fallback") for c in ch2)


def test_headings_case_sensitive_and_inline():
    text = "AUTOS Y VISTOS: algo. Considerando que el juez decide. Fs. 12. Y CONSIDERANDO: Que X.\nPor ello, RESUELVO: I.- Rechazar."
    heads, strict = detect_headings(text)
    assert [h.label for h in heads] == ["vistos", "considerando", "resuelvo"]
    assert not strict
    secs = build_sections(text, heads)
    assert [s.label for s in secs] == ["vistos", "considerando", "resuelvo"]


def test_uppercase_document_uses_strict_mode():
    text = "AUTOS Y VISTOS:\nEL TRIBUNAL CONSIDERANDO QUE LA PENA ES JUSTA.\nCONSIDERANDO:\nQUE SE RESUELVE ALGO.\nRESUELVO:\nRECHAZAR."
    heads, strict = detect_headings(text)
    assert strict
    assert [h.label for h in heads] == ["vistos", "considerando", "resuelvo"]


def test_config_errors(common):
    with pytest.raises(ConfigError):
        KINDS["fixed_tokens"]("f", {"window_tokens": 999}, common, 1)
    with pytest.raises(ConfigError):
        KINDS["semantic_dp"]("s", {"inexistente": 1}, common, 1)
    with pytest.raises(ConfigError):
        KINDS["semantic_dp"]("s", {"min_sentences": 9, "ideal_sentences": 3}, common, 1)


def test_validator_detects_faults(tok, emb, common):
    from dataclasses import replace

    doc = make_doc(seed=1, n_sent=30)
    view = make_view(doc, tok, emb, common)
    st = KINDS["sentence_pack"]("p", {"target_tokens": 60}, common, 1)
    ch = finalize(view, st.chunk(view), "p")
    kw = dict(strategy="p", max_tokens=common.max_tokens, hard_limit=10_000, tokenizer=tok,
              coverage_mode="exact", allows_overlap=False)
    assert not validate_chunks(doc, ch, **kw)
    codes = {i.code for i in validate_chunks(doc, ch[1:], **kw)}
    assert "COVERAGE" in codes
    codes = {i.code for i in validate_chunks(doc, ch + [ch[0]], **kw)}
    assert "DUPLICATION" in codes or "DUP_ID" in codes
    bad = [replace(ch[0], text=ch[0].text + "x")] + ch[1:]
    assert "TEXT_MISMATCH" in {i.code for i in validate_chunks(doc, bad, **kw)}
    tiny = {**kw, "max_tokens": 5}
    assert "TOKEN_CAP" in {i.code for i in validate_chunks(doc, ch, **tiny)}
