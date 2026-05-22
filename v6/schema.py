"""v6/schema.py — Live database introspection.

The schema is read from the database itself, never hand-maintained:
  - table / column names and types via PRAGMA (SQLite) or information_schema,
  - human column descriptions from the `data_catalog` table.

From that it derives the **join map**: any table carrying a `location_id`
column must JOIN `dim_location` to be filtered by a wilaya name. That single
rule replaces the old, wrong assumption that `global_revenue` had a `wilaya`
column — global_revenue carries `location_id` like every other metric table,
so it is treated like every other metric table. This is the structural fix
for the `no such column: wilaya` failures.
"""

from __future__ import annotations
import sqlite3

from .config import V6Config


def db_connect():
    """Open a connection to the active backend (SQLite on Colab, MySQL local)."""
    if V6Config.USE_SQLITE:
        conn = sqlite3.connect(V6Config.sqlite_path())
        conn.row_factory = sqlite3.Row
        return conn
    import mysql.connector
    return mysql.connector.connect(
        host=V6Config.MYSQL_HOST, port=V6Config.MYSQL_PORT,
        user=V6Config.MYSQL_USER, password=V6Config.MYSQL_PASSWORD,
        database=V6Config.MYSQL_DB,
        connection_timeout=V6Config.SQL_TIMEOUT_S)


class DBSchema:
    """Introspected schema + the derived join map. Built once, then cached."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {}     # table -> [{name, type}]
        self.descriptions: dict[tuple, str] = {}    # (table, col) -> description
        self.join_map: dict[str, str] = {}          # table -> fk column name
        self.date_range: tuple[str, str] | None = None
        self._introspect()

    # ── introspection ────────────────────────────────────────────────────
    def _introspect(self) -> None:
        conn = db_connect()
        try:
            if V6Config.USE_SQLITE:
                self._introspect_sqlite(conn)
            else:
                self._introspect_mysql(conn)
            self._load_descriptions(conn)
            self._find_date_range(conn)
        finally:
            conn.close()
        # Derive the join map: any table (other than dim_location) with a
        # location_id column needs a dim_location join for wilaya filters.
        for table, cols in self.tables.items():
            names = [c["name"] for c in cols]
            if table != "dim_location" and "location_id" in names:
                self.join_map[table] = "location_id"

    def _introspect_sqlite(self, conn) -> None:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table,) in cur.fetchall():
            if table.startswith("sqlite_"):
                continue
            cur.execute(f"PRAGMA table_info('{table}')")
            self.tables[table] = [
                {"name": r[1], "type": (r[2] or "").upper()}
                for r in cur.fetchall()]

    def _introspect_mysql(self, conn) -> None:
        cur = conn.cursor()
        cur.execute(
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
            "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = %s "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION", (V6Config.MYSQL_DB,))
        for table, col, dtype in cur.fetchall():
            self.tables.setdefault(table, []).append(
                {"name": col, "type": (dtype or "").upper()})

    def _load_descriptions(self, conn) -> None:
        """data_catalog is the database documenting its own columns."""
        try:
            cur = conn.cursor()
            cur.execute("SELECT table_name, column_name, description "
                        "FROM data_catalog")
            for row in cur.fetchall():
                self.descriptions[(row[0], row[1])] = row[2]
        except Exception:  # noqa: BLE001 — data_catalog is optional
            pass

    def _find_date_range(self, conn) -> None:
        for table in ("prepaid_kpi", "global_revenue", "postpaid_kpi"):
            if table not in self.tables:
                continue
            try:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT MIN(week_start), MAX(week_start) FROM `{table}`")
                lo, hi = cur.fetchone()
                if lo and hi:
                    self.date_range = (str(lo), str(hi))
                    return
            except Exception:  # noqa: BLE001 — table may lack week_start
                continue

    # ── queries ──────────────────────────────────────────────────────────
    def all_tables(self) -> list[str]:
        return list(self.tables)

    def metric_tables(self) -> list[str]:
        return sorted(self.join_map)

    def column_names(self, table: str) -> list[str]:
        return [c["name"] for c in self.tables.get(table, [])]

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_column(self, table: str, col: str) -> bool:
        return col in self.column_names(table)

    def needs_location_join(self, table: str) -> bool:
        return table in self.join_map

    def numeric_columns(self, table: str) -> list[str]:
        num = ("INT", "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")
        return [c["name"] for c in self.tables.get(table, [])
                if any(k in c["type"] for k in num)]

    # ── router prompt ─────────────────────────────────────────────────────
    def prompt(self) -> str:
        """The schema block injected into the router SLM's prompt."""
        backend = "sqlite" if V6Config.USE_SQLITE else V6Config.MYSQL_DB
        lines = [f"Database schema (`{backend}`):", ""]
        for table in sorted(self.tables):
            cols = [c["name"] for c in self.tables[table] if c["name"] != "id"]
            lines.append(f"  {table}({', '.join(cols)})")
        if self.join_map:
            lines += [
                "",
                "JOIN RULE — read carefully:",
                "  Metric tables (" + ", ".join(self.metric_tables()) + ")",
                "  store location as `location_id`, NOT as a wilaya name.",
                "  To filter or group by a wilaya you MUST join dim_location:",
                "    JOIN dim_location ON <table>.location_id "
                "= dim_location.location_id",
                "  then use dim_location.wilaya. No metric table has its own "
                "`wilaya` column.",
            ]
        if self.date_range:
            lines += [
                "",
                f"Data covers weekly snapshots from {self.date_range[0]} to "
                f"{self.date_range[1]}; resolve relative time expressions "
                f"against {self.date_range[1]}.",
                "TIME RULE: The date column is `week_start` (NOT `time`, `date`, "
                "`timestamp`, or `period`). Always use `week_start` for date filters.",
            ]
        return "\n".join(lines)


_schema: DBSchema | None = None


def get_db_schema(refresh: bool = False) -> DBSchema:
    global _schema
    if _schema is None or refresh:
        _schema = DBSchema()
    return _schema
