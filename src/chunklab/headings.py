"""Deteccion de encabezados de sentencia (AUTOS Y VISTOS / CONSIDERANDO / RESUELVO).

Criterios
  * Coincidencia SENSIBLE a mayusculas: los encabezados estan en mayusculas; asi
    "considerando que..." en prosa no se confunde con un encabezado.
  * Posicion valida: comienzo de linea (con numeracion opcional), o inline tras
    signo de puntuacion de cierre ('. Y CONSIDERANDO:') o tras 'Por ello,'.
  * Documentos casi todo en mayusculas: modo estricto (solo comienzo de linea y
    seguido de ':' o fin de linea).
Esta deteccion es una HEURISTICA: debe validarse contra anotacion manual
(`chunklab diagnose-headings`)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .core import Span

HEADINGS_VERSION = "1.0"

_PATTERNS: tuple[tuple[str, str], ...] = (
    ("considerando", r"VISTOS\s+Y\s+CONSIDERANDO|Y\s+CONSIDERANDO(?:\s+QUE)?|CONSIDERANDO"),
    ("vistos", r"AUTOS\s+Y\s+VISTOS|VISTOS?"),
    ("resuelvo", r"SE\s+RESUELVE|RESUELVO|RESUELVE|FALLO|FALLA"),
)
_WORD = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_"
_HEADING_RE = re.compile(
    rf"(?<![{_WORD}])(?:"
    + "|".join(f"(?P<{label}>{pat})" for label, pat in _PATTERNS)
    + rf")(?![{_WORD}])(?P<tail>[ \t]*(?::|\.-|-|\.)?)"
)
_NUMBERING = re.compile(r"[ \t]*(?:\(?[0-9IVXivx]{1,4}[.)\-]*[ \t]*)?")
_LEAD_IN = re.compile(
    r"(?i)(?:por\s+ello|por\s+lo\s+tanto|en\s+consecuencia|por\s+todo\s+ello)[ \t]*,?[ \t]*$"
)


@dataclass(frozen=True)
class Heading:
    start: int
    end: int
    label: str  # 'vistos' | 'considerando' | 'resuelvo'
    text: str


@dataclass(frozen=True)
class Section:
    label: str  # 'preambulo' | 'vistos' | 'considerando' | 'resuelvo'
    span: Span


def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _at_line_start(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    return _NUMBERING.fullmatch(text[line_start:start]) is not None


def _inline_ok(text: str, start: int) -> bool:
    j = start - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    if j < 0 or text[j] in ".:;)\n–—-":
        return True
    return _LEAD_IN.search(text[max(0, start - 40) : start]) is not None


def detect_headings(text: str, *, strict_ratio: float = 0.6) -> tuple[list[Heading], bool]:
    """Devuelve (encabezados ordenados, modo_estricto)."""
    strict = uppercase_ratio(text) > strict_ratio
    found: list[Heading] = []
    for m in _HEADING_RE.finditer(text):
        label = next(lbl for lbl, _ in _PATTERNS if m.group(lbl) is not None)
        start = m.start()
        if strict:
            nxt = text[m.end() : m.end() + 1]
            if not (_at_line_start(text, start) and (":" in m.group("tail") or nxt in ("", "\n"))):
                continue
        elif not (_at_line_start(text, start) or _inline_ok(text, start)):
            continue
        found.append(Heading(start, m.end(), label, m.group(0).strip()))
    return found, strict


def build_sections(text: str, headings: list[Heading]) -> list[Section]:
    """Secciones contiguas: preambulo (si existe) y una por encabezado."""
    if not headings:
        return []
    bounds = [h.start for h in headings] + [len(text)]
    sections: list[Section] = []

    def add(label: str, a: int, b: int) -> None:
        seg = text[a:b]
        s = seg.strip()
        if s:
            left = len(seg) - len(seg.lstrip())
            sections.append(Section(label, Span(a + left, a + left + len(s))))

    add("preambulo", 0, headings[0].start)
    for h, end in zip(headings, bounds[1:]):
        add(h.label, h.start, end)
    return sections
