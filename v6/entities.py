"""v6/entities.py — Deterministic entity resolution.

The router SLM extracts entity *mentions* ("Algiers", "last month"); this
module turns them into values that actually exist in the database.

`dim_location` stores wilaya names in French (`Alger`, `Béjaïa`). Users say
them in English (`Algiers`), in Arabic (`الجزائر`), with no accents
(`Bejaia`), or with typos. Resolution here is accent-insensitive, fuzzy as a
last resort, and alias-aware via **data/wilaya_aliases.json** — that file is
the editable source of truth for variants. No alias is hard-coded in Python
anymore: drop a new spelling into the JSON and it works on the next start.

`dim_location` is commune-level (~25 communes per wilaya), so we resolve
each mention to BOTH the canonical name AND the full list of `location_id`
values that belong to that wilaya. SQL then filters with
`WHERE location_id IN (id1, id2, ...)` — wilaya-level aggregation with no
JOIN required.
"""

from __future__ import annotations
import json
import os
import re
import unicodedata
from difflib import get_close_matches

from .config import V6Config
from .schema import db_connect


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
        self.wilaya_to_ids: dict[str, list[int]] = self._load_wilayas()
        self.wilayas: list[str] = list(self.wilaya_to_ids.keys())
        self._known: set[str] = set(self.wilayas)
        self._index: dict[str, str] = {}             # normalized form -> canonical
        for w in self.wilayas:
            self._index[_norm(w)] = w
        for canon, aliases in _load_aliases().items():
            if canon in self._known:
                for alias in aliases:
                    self._index[_norm(alias)] = canon
        self._max_words = max(
            (len(_norm(w).split()) for w in self.wilayas), default=1)

    def _load_wilayas(self) -> dict[str, list[int]]:
        """Return {canonical wilaya name → sorted list of its location_ids}."""
        out: dict[str, list[int]] = {}
        try:
            conn = db_connect()
            cur = conn.cursor()
            cur.execute("SELECT wilaya, location_id FROM dim_location "
                        "WHERE wilaya IS NOT NULL "
                        "ORDER BY wilaya, location_id")
            for wilaya, loc_id in cur.fetchall():
                out.setdefault(wilaya, []).append(int(loc_id))
            conn.close()
        except Exception:  # noqa: BLE001 — resolver degrades to empty
            return {}
        return out

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

    def resolve_many(self, names: list) -> dict:
        resolved: list[str] = []
        unresolved: list[str] = []
        for n in names or []:
            hit = self.resolve_wilaya(str(n))
            if hit and hit not in resolved:
                resolved.append(hit)
            elif not hit:
                unresolved.append(str(n))
        return {"resolved": resolved, "unresolved": unresolved}

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

    def ids_for(self, wilaya: str) -> list[int]:
        """Every commune location_id that belongs to a canonical wilaya."""
        return list(self.wilaya_to_ids.get(wilaya, []))

    # ── convenience ──────────────────────────────────────────────────────
    def resolve_all(self, query: str, router_filters: dict | None,
                    max_date: str | None = None) -> dict:
        """Combine router-supplied filters with a direct query scan.

        Returns `wilayas` (canonical French spellings) and `wilaya_ids_map`
        ({wilaya: [all its commune location_ids]}). Time and segment are
        no longer resolved here — the router SLM already extracts those
        from the query, and the SQL-gen SLM reads them from the routing
        object plus the schema's date-range hint, so a Python substring
        list is just brittle duplication. The `max_date` argument is kept
        for backwards compatibility but ignored."""
        del max_date  # noqa: ignored — SLM handles time math
        rf = router_filters or {}
        from_router = self.resolve_many(rf.get("wilayas", []))
        scanned = self.scan_query(query)
        wilayas = list(dict.fromkeys(from_router["resolved"] + scanned))
        ids_map = {w: self.ids_for(w) for w in wilayas}
        return {
            "wilayas": wilayas,
            "wilaya_ids_map": ids_map,
            "unresolved_wilayas": from_router["unresolved"],
        }


_resolver: Resolver | None = None


def get_resolver() -> Resolver:
    global _resolver
    if _resolver is None:
        _resolver = Resolver()
    return _resolver
