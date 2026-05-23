"""v6/entities.py — Deterministic entity resolution.

The router SLM extracts entity *mentions* ("Algiers", "last month"); this
module turns them into values that actually exist in the database.

`dim_location` stores wilaya names in French (`Alger`, `Béjaïa`). Users say
them in English (`Algiers`), in Arabic (`الجزائر`), with no accents
(`Bejaia`), or with typos. Resolution here is accent-insensitive, fuzzy as a
last resort, and alias-aware via **data/wilaya_aliases.json** — that file is
the editable source of truth for variants. No alias is hard-coded in Python
anymore: drop a new spelling into the JSON and it works on the next start.

We resolve to BOTH the canonical name and the `location_id`. SQL filtering
uses the id (`WHERE location_id IN (16, 25)`) because integers can't be
misspelled; display can still JOIN dim_location for the human-readable name.
"""

from __future__ import annotations
import datetime as _dt
import json
import os
import re
import unicodedata
from difflib import get_close_matches

from .config import V6Config
from .schema import db_connect

_SEGMENTS = ("prepaid", "postpaid", "b2b", "b2c")


def _norm(s: str) -> str:
    """Lowercase, drop accents / apostrophes / punctuation, collapse spaces."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", "").replace("’", "").replace("`", "")
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z0-9؀-ۿ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_aliases() -> dict:
    """Read data/wilaya_aliases.json — {canonical: [aliases]}. Optional."""
    if not os.path.isfile(V6Config.WILAYA_ALIASES_PATH):
        return {}
    try:
        with open(V6Config.WILAYA_ALIASES_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, list)}
    except Exception:  # noqa: BLE001 — broken file degrades to empty
        return {}


class Resolver:
    """Maps free-text entity mentions onto real database values."""

    def __init__(self):
        rows = self._load_wilayas()                  # [(location_id, wilaya)]
        self.wilayas: list[str] = [w for _, w in rows]
        self.wilaya_to_id: dict[str, int] = {w: int(i) for i, w in rows}
        self._index: dict[str, str] = {}             # normalized form -> canonical
        for w in self.wilayas:
            self._index[_norm(w)] = w
        for canon, aliases in _load_aliases().items():
            if canon in self.wilaya_to_id:
                for alias in aliases:
                    self._index[_norm(alias)] = canon
        self._max_words = max(
            (len(_norm(w).split()) for w in self.wilayas), default=1)

    def _load_wilayas(self) -> list[tuple[int, str]]:
        try:
            conn = db_connect()
            cur = conn.cursor()
            cur.execute("SELECT location_id, wilaya FROM dim_location "
                        "WHERE wilaya IS NOT NULL ORDER BY wilaya")
            out = [(r[0], r[1]) for r in cur.fetchall()]
            conn.close()
            return out
        except Exception:  # noqa: BLE001 — resolver degrades to empty
            return []

    # ── wilaya resolution ────────────────────────────────────────────────
    def resolve_wilaya(self, name: str) -> str | None:
        """One mention → its canonical DB value, or None if unknown."""
        key = _norm(name)
        if not key:
            return None
        if key in self._index:
            return self._index[key]
        near = get_close_matches(key, list(self._index), n=1, cutoff=0.86)
        return self._index[near[0]] if near else None

    def wilaya_id(self, name: str) -> int | None:
        """One mention → its location_id, or None if unknown."""
        canon = self.resolve_wilaya(name)
        return self.wilaya_to_id.get(canon) if canon else None

    def resolve_many(self, names: list) -> dict:
        resolved: list[str] = []
        ids: list[int] = []
        unresolved: list[str] = []
        for n in names or []:
            hit = self.resolve_wilaya(str(n))
            if hit and hit not in resolved:
                resolved.append(hit)
                wid = self.wilaya_to_id.get(hit)
                if wid is not None:
                    ids.append(wid)
            elif not hit:
                unresolved.append(str(n))
        return {"resolved": resolved, "ids": ids, "unresolved": unresolved}

    def scan_query(self, query: str) -> list[str]:
        """Find wilaya names directly in free text — backs up the router.

        Exact-index only (no fuzzy) so ordinary words can't false-match.
        Matches the longest n-gram first so 'Tizi Ouzou' beats 'Tizi'.
        """
        toks = _norm(query).split()
        found: list[str] = []
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(self._max_words, len(toks) - i), 0, -1):
                gram = " ".join(toks[i:i + n])
                if gram in self._index:
                    hit = (self._index[gram], n)
                    break
            if hit:
                if hit[0] not in found:
                    found.append(hit[0])
                i += hit[1]
            else:
                i += 1
        return found

    # ── time + segment ───────────────────────────────────────────────────
    def resolve_time(self, query: str, max_date: str | None) -> dict | None:
        """Relative time expressions, resolved against the last data week."""
        if not max_date:
            return None
        try:
            hi = _dt.date.fromisoformat(max_date[:10])
        except ValueError:
            return None
        q = query.lower()

        def span(days: int) -> dict:
            return {"start": (hi - _dt.timedelta(days=days)).isoformat(),
                    "end": hi.isoformat()}

        m = re.search(r"last\s+(\d+)\s+weeks?", q)
        if m:
            return span(7 * int(m.group(1)))
        m = re.search(r"last\s+(\d+)\s+months?", q)
        if m:
            return span(30 * int(m.group(1)))
        if any(k in q for k in ("last quarter", "past quarter", "previous quarter")):
            return span(90)
        if any(k in q for k in ("last month", "past month")):
            return span(30)
        if any(k in q for k in ("last week", "past week")):
            return span(7)
        if any(k in q for k in ("recently", "lately", "recent weeks")):
            return span(30)
        if any(k in q for k in ("this year", "year to date", "ytd")):
            return {"start": f"{hi.year}-01-01", "end": hi.isoformat()}
        return None

    def resolve_segment(self, query: str) -> str | None:
        q = query.lower()
        for seg in _SEGMENTS:
            if seg in q:
                return seg
        return None

    # ── convenience ──────────────────────────────────────────────────────
    def resolve_all(self, query: str, router_filters: dict | None,
                    max_date: str | None) -> dict:
        """Combine router-supplied filters with a direct query scan."""
        rf = router_filters or {}
        from_router = self.resolve_many(rf.get("wilayas", []))
        scanned = self.scan_query(query)
        wilayas = list(dict.fromkeys(from_router["resolved"] + scanned))
        ids = [self.wilaya_to_id[w] for w in wilayas if w in self.wilaya_to_id]
        return {
            "wilayas": wilayas,
            "wilaya_ids": ids,
            "unresolved_wilayas": from_router["unresolved"],
            "segment": self.resolve_segment(query) or rf.get("segment"),
            "time_range": self.resolve_time(query, max_date),
        }


_resolver: Resolver | None = None


def get_resolver() -> Resolver:
    global _resolver
    if _resolver is None:
        _resolver = Resolver()
    return _resolver
