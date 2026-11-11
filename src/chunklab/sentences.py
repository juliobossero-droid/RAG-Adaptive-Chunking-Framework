"""Segmentador de oraciones para castellano juridico, con offsets exactos.

Reglas principales
  * Un salto de parrafo (linea en blanco) siempre separa oraciones, y tambien un
    numerador de lista al inicio de linea (I.-, II., 1.).
  * Un salto de linea simple NO separa (los .txt suelen tener corte duro de linea).
  * . ! ? ... seguidos de espacio y de mayuscula/digito separan, salvo que el punto
    pertenezca a una abreviatura juridica (art., inc., fs., Dr.), a siglas
    (C.P.P., S.S.), a una inicial de nombre (J. Perez) o a un numerador de lista
    al comienzo de linea (1., II.).
  * Se aceptan fronteras forzadas (inicio de encabezados juridicos).
  * Las oraciones mas largas que el tope de tokens se subdividen (atomizacion),
    para que ningun chunk pueda truncarse en el embedding.
"""

from __future__ import annotations

import re
from typing import Iterable

from .core import Span
from .tokenization import TokenIndex

SEGMENTER_VERSION = "1.0"

DEFAULT_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "art", "arts", "inc", "incs", "ap", "apart", "cap", "cód", "cod", "cf", "cfr",
        "dr", "dra", "dres", "sr", "sra", "sres", "srta", "lic", "ing", "arq", "prof",
        "mag", "esc", "fdo", "nº", "n°", "nro", "nros", "núm", "num", "fs", "fj", "ff",
        "ss", "sig", "sigs", "pág", "pags", "págs", "pp", "vol", "tít", "tit", "sec",
        "secr", "dep", "dpto", "depto", "dto", "av", "avda", "gral", "cnel", "tte",
        "cmte", "vs", "ej", "v.gr", "vgr", "expte", "exp", "expdte", "juzg", "trib",
        "cám", "cam", "fisc", "def", "ofic", "prov", "bs", "as", "adm", "mun", "nac",
        "fed", "pcia", "dec", "res", "resol", "reg", "ibid", "ib", "cit", "op", "loc",
    }
)

_PARAGRAPH_BREAK = re.compile(r"\n[ \t\f\v]*\n\s*")
_TERMINAL = re.compile(
    r"(?P<punct>[.!?…]+)(?P<close>[\"'”’»)\]]*)(?P<ws>\s+)"
    r"(?=[¿¡\"“«(\[]?[A-ZÁÉÍÓÚÜÑ0-9])"
)
_LIST_MARKER = re.compile(
    r"(?m)^[ \t]*((?:[IVX]{1,5}|\d{1,2})(?:\.-|\.|\)))[ \t]+(?=[A-ZÁÉÍÓÚÜÑ¿¡\"“«])"
)
_LAST_WORD = re.compile(r"[^\s(\[¿¡\"“«]+$")
_INITIALS = re.compile(r"^(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]\.)+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]$")
_SINGLE_CAPITAL = re.compile(r"^[A-ZÁÉÍÓÚÜÑ]$")
_ROMAN = re.compile(r"^[IVXLCDM]{1,6}$")
_SMALL_NUMBER = re.compile(r"^\d{1,2}$")
_SOFT_CUT = re.compile(r"[;:]\s+|\n+|,\s+")


def _is_false_boundary(text: str, punct_start: int, punct: str, abbreviations: frozenset[str]) -> bool:
    """True si el punto NO cierra oracion."""
    if punct != ".":
        return False  # ! ? y ... siempre cierran
    window = text[max(0, punct_start - 48) : punct_start]
    m = _LAST_WORD.search(window)
    if not m:
        return False
    word = m.group(0)
    w_start = punct_start - len(word)
    if word.lower() in abbreviations:
        return True
    if _INITIALS.match(word) or _SINGLE_CAPITAL.match(word):
        return True
    if _ROMAN.match(word) or _SMALL_NUMBER.match(word):
        line_start = text.rfind("\n", 0, w_start) + 1
        if not text[line_start:w_start].strip():
            return True  # numerador de lista al comienzo de linea
    return False


def segment_sentences(
    text: str,
    *,
    abbreviations: frozenset[str] = DEFAULT_ABBREVIATIONS,
    forced_boundaries: Iterable[int] = (),
) -> list[Span]:
    """Particion del texto en oraciones (spans recortados, sin espacios en los bordes).

    Propiedad garantizada: todo caracter no blanco del texto pertenece a exactamente
    una oracion."""
    n = len(text)
    cuts = {0, n}
    for m in _PARAGRAPH_BREAK.finditer(text):
        cuts.add(m.end())
    for m in _LIST_MARKER.finditer(text):  # items numerados al inicio de linea (I.-, 2., a))
        cuts.add(m.start(1))
    for m in _TERMINAL.finditer(text):
        if not _is_false_boundary(text, m.start("punct"), m.group("punct"), abbreviations):
            cuts.add(m.end("ws"))
    for b in forced_boundaries:
        if 0 < int(b) < n:
            cuts.add(int(b))
    ordered = sorted(cuts)
    spans: list[Span] = []
    for a, b in zip(ordered, ordered[1:]):
        seg = text[a:b]
        stripped = seg.strip()
        if not stripped:
            continue
        left = len(seg) - len(seg.lstrip())
        spans.append(Span(a + left, a + left + len(stripped)))
    return spans


def _trim(text: str, a: int, b: int) -> Span | None:
    seg = text[a:b]
    stripped = seg.strip()
    if not stripped:
        return None
    left = len(seg) - len(seg.lstrip())
    return Span(a + left, a + left + len(stripped))


def split_oversized(
    sentences: list[Span], token_index: TokenIndex, text: str, max_tokens: int
) -> tuple[list[Span], dict[str, int]]:
    """Atomizacion: subdivide oraciones de mas de `max_tokens` tokens.

    Primero intenta cortar en ; : salto de linea o coma (corte 'suave'); si no hay
    candidato, corta duro en el limite de token numero `max_tokens`. Devuelve las
    nuevas oraciones y contadores {'soft': n, 'hard': n, 'oversized': n}."""
    out: list[Span] = []
    stats = {"soft": 0, "hard": 0, "oversized": 0}
    for s in sentences:
        if token_index.count(s.start, s.end) <= max_tokens:
            out.append(s)
            continue
        stats["oversized"] += 1
        cands = [s.start + m.end() for m in _SOFT_CUT.finditer(text[s.start : s.end])]
        cur = s.start
        while cur < s.end:
            if token_index.count(cur, s.end) <= max_tokens:
                piece = _trim(text, cur, s.end)
                if piece:
                    out.append(piece)
                break
            best = None
            for c in cands:
                if c <= cur or c >= s.end:
                    continue
                if token_index.count(cur, c) <= max_tokens:
                    best = c
                else:
                    break
            if best is not None:
                stats["soft"] += 1
                cut = best
            else:
                stats["hard"] += 1
                first = token_index.first_at_or_after(cur)
                cut = token_index.starts[first + max_tokens]
            piece = _trim(text, cur, cut)
            if piece:
                out.append(piece)
            cur = cut
    return out, stats
