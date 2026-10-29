"""Searchable medicine / manufacturer catalogue used for manual-entry autocomplete.

The catalogue is a convenience list for typing faster. A match in this list is
NOT evidence that a medicine is genuine.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import config


@dataclass(frozen=True)
class CatalogueEntry:
    medicine: str
    manufacturer: str
    composition: str
    strength: str
    form: str


@lru_cache(maxsize=4)
def load_catalogue(path: str | None = None) -> tuple[CatalogueEntry, ...]:
    p = Path(path) if path else config.CATALOGUE_PATH
    if not p.exists():
        return tuple()
    with p.open(encoding="utf-8") as fh:
        rows = [
            CatalogueEntry(**{k: (r.get(k) or "").strip() for k in
                              ("medicine", "manufacturer", "composition", "strength", "form")})
            for r in csv.DictReader(fh)
            if (r.get("medicine") or "").strip()
        ]
    return tuple(rows)


@lru_cache(maxsize=4)
def load_manufacturers(path: str | None = None) -> tuple[str, ...]:
    names: list[str] = []
    p = Path(path) if path else config.DATA_DIR / "manufacturers.csv"
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            names += [(r.get("manufacturer") or "").strip() for r in csv.DictReader(fh)]
    names += [e.manufacturer for e in load_catalogue()]
    return tuple(sorted({n for n in names if n}, key=str.lower))


def _score(candidate: str, query: str) -> int | None:
    """Lower score = better match. None = no match."""
    c, q = candidate.lower(), query.lower().strip()
    if not q:
        return 3
    if c.startswith(q):
        return 0
    if any(word.startswith(q) for word in c.replace("-", " ").split()):
        return 1
    if q in c:
        return 2
    return None


def suggest(query: str, options: list[str] | tuple[str, ...], limit: int = 8) -> list[str]:
    scored = []
    for opt in dict.fromkeys(options):  # de-duplicate, keep order
        s = _score(opt, query)
        if s is not None:
            scored.append((s, opt.lower(), opt))
    scored.sort()
    return [opt for _, _, opt in scored[:limit]]


def suggest_manufacturers(query: str, limit: int = 8) -> list[str]:
    return suggest(query, load_manufacturers(), limit)


def suggest_medicines(query: str, limit: int = 8) -> list[str]:
    return suggest(query, [e.medicine for e in load_catalogue()], limit)


def suggest_compositions(query: str, limit: int = 8) -> list[str]:
    return suggest(query, sorted({e.composition for e in load_catalogue()}), limit)


def lookup_medicine(name: str) -> CatalogueEntry | None:
    name_l = (name or "").strip().lower()
    for e in load_catalogue():
        if e.medicine.lower() == name_l:
            return e
    return None


def medicines_for_manufacturer(manufacturer: str) -> list[str]:
    m = (manufacturer or "").strip().lower()
    return [e.medicine for e in load_catalogue() if e.manufacturer.lower() == m]
