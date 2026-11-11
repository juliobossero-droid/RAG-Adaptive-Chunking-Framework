from chunklab.sentences import segment_sentences, split_oversized
from chunklab.tokenization import RegexTokenizerAdapter, TokenIndex


def texts(s):
    return [s[x.start : x.end] for x in segment_sentences(s)]


def test_abbreviations_do_not_split():
    s = "Conforme el art. 13 del C.P.P. se resuelve. El Dr. Perez firmo fs. 12 el escrito."
    out = texts(s)
    assert len(out) == 2
    assert out[0].endswith("resuelve.") and out[1].startswith("El Dr.")


def test_initials_and_list_markers():
    s = "Declaro J. Perez ante el tribunal. Luego:\nII. Se ordena la medida. Fin del acto."
    out = texts(s)
    assert any(o.startswith("II. Se ordena") for o in out)
    assert any("J. Perez" in o for o in out)


def test_question_exclamation_and_paragraphs():
    s = "Que decide? Se rechaza! Nuevo parrafo sin punto\n\nOtro parrafo."
    assert len(texts(s)) == 4


def test_every_nonwhitespace_char_in_exactly_one_sentence():
    s = "Uno dos tres. Cuatro cinco, seis; siete.\n\nOcho nueve. Diez!"
    spans = segment_sentences(s)
    cover = [0] * len(s)
    for sp in spans:
        for i in range(sp.start, sp.end):
            cover[i] += 1
    for ch, c in zip(s, cover):
        assert c == (0 if ch.isspace() and c == 0 else c)
        if not ch.isspace():
            assert c == 1


def test_forced_boundaries():
    s = "CONSIDERANDO: Que el tribunal resuelve sin punto intermedio"
    spans = segment_sentences(s, forced_boundaries=[14])
    assert len(spans) == 2


def test_split_oversized_respects_cap_and_covers():
    tok = RegexTokenizerAdapter()
    s = " ".join(f"palabra{i}" for i in range(200)) + "."
    ti = TokenIndex(tok.offsets(s))
    spans = segment_sentences(s)
    out, stats = split_oversized(spans, ti, s, 50)
    assert all(ti.count(x.start, x.end) <= 50 for x in out)
    assert stats["hard"] > 0
    joined = "".join(s[x.start : x.end] for x in out).replace(" ", "")
    assert joined == s.replace(" ", "")
