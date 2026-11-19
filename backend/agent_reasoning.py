"""Curation memory: the reason the curator model gives each pick, and how it feeds back.

When :func:`backend.generator.generate_playlist_stream` matches the curator's
selections against the library, each pick comes with a one-sentence reason
("why this track fits", requested in :data:`backend.generator.GENERATION_SYSTEM`).
Those reasons are kept in their own SQLite file — ``data/agent_song_reasoning.db`` —
separate from the library cache DB. This module is **not** a second track cache:
its rows are the agent's decision history ("this run chose this track because ..."),
keyed by Plex ``rating_key``, so nothing here answers a library query.

The history serves two purposes:

* **Reuse** — :func:`get_previous_reasons` replays a track's most recent stored
  reason next to its candidate line, so an argument the agent already made in an
  earlier run does not have to be re-derived by the model.
* **Processing** — :func:`record_run` mines every reason into normalized tags
  (genre keywords and decade hints through :mod:`backend.prompt_classifier`, so
  the tags share vocabulary with the hints the filters and the pre-rank compute),
  and the "reasoning affinity" component of :func:`backend.generator.pre_rank_tracks`
  ranks candidates by how much those tags overlap with the current request — the
  agent's own selection reasons thus enrich the filtering that decides which
  tracks reach the AI step.

The DB path is derived from module location exactly like :mod:`backend.library_cache`
(:data:`DATA_DIR` is ``Path(__file__).parent.parent / "data"``, so the file is repo-root
``data/agent_song_reasoning.db`` regardless of cwd), and :data:`DB_PATH` is bound **at
import** — an isolated test must repatch :data:`DB_PATH` too, not just :data:`DATA_DIR`
(the same trap that once let a fixture overwrite the real cache). ``*.db`` is gitignored,
so the file is machine state: never quote its row counts as a repo fact.
"""

import logging
import secrets
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from backend import prompt_classifier

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "agent_song_reasoning.db"

# Budgets on stored and replayed reason text, so model prose can neither bloat the
# DB nor one prompt line.
MAX_REASON_CHARS = 280  # Longest reason kept per stored pick
MAX_PAST_REASON_CHARS = 160  # Longest reason replayed beside a candidate line
MAX_TRACKS_WITH_PAST_REASON = 400  # Candidate lines that may replay a reason
MAX_TAGS_PER_REASON = 12  # Mined tags one reason may contribute
MAX_TAG_CHARS = 48  # Longest mined tag
MAX_STATUS_TAGS = 40  # Tags :func:`reasoning_status` reports
QUERY_PARAM_CHUNK = 900  # rating keys per IN-clause (under SQLite's 9999-param cap)

_schema_initialized = False
_schema_lock = threading.Lock()


def _clean_tag(tag: str) -> str:
    """Normalize one mined tag: collapse case/whitespace and cap its length."""
    return " ".join((tag or "").split()).lower()[:MAX_TAG_CHARS]


def _mine_tags(reason: str) -> list[str]:
    """Mined vocabulary of one reason: keywords it names plus what they expand to.

    Kept in the same vocabulary :func:`backend.prompt_classifier.extract_genre_hints`
    produces for a request, so comparing a track's tags with a request's hints is a
    plain set overlap and no new keyword list is invented here.
    """
    lowered = (reason or "").lower()
    tags: set[str] = set()
    for keyword, genres in prompt_classifier.extract_genre_hint_groups(lowered).items():
        tags.add(_clean_tag(keyword))
        tags.update(_clean_tag(genre) for genre in genres)
    tags.update(_clean_tag(decade) for decade in prompt_classifier.extract_decade_hints(lowered))
    tags.discard("")
    return sorted(tags)[:MAX_TAGS_PER_REASON]


def _connect() -> sqlite3.Connection:
    """Open the reasoning DB with the same pragmas as the library cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create the three reasoning tables (idempotent).

    ``runs`` records one row per generation run (context: the request, the
    playlist title, the cache ``results`` id when the run saved one);
    ``selections`` the per-track reason, PK ``(run_id, rating_key)``; ``tags``
    the mined vocabulary, PK ``(rating_key, tag)`` so a reused tag costs no row.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'prompt',
            prompt TEXT NOT NULL DEFAULT '',
            playlist_title TEXT NOT NULL DEFAULT '',
            result_id TEXT NOT NULL DEFAULT '',
            track_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS selections (
            run_id TEXT NOT NULL,
            rating_key TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, rating_key)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_keys ON selections(rating_key)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tags (
            rating_key TEXT NOT NULL,
            tag TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (rating_key, tag)
        )"""
    )


def ensure_db_initialized() -> sqlite3.Connection:
    """Open the reasoning DB, creating its schema once per process."""
    global _schema_initialized
    conn = _connect()
    if not _schema_initialized:
        with _schema_lock:
            if not _schema_initialized:
                _init_tables(conn)
                _schema_initialized = True
    return conn


def record_run(
    selections: list[tuple[str, str]],
    prompt: str = "",
    playlist_title: str = "",
    result_id: str = "",
    source: str = "prompt",
) -> int:
    """Store one generation run's per-track selection reasons; rows written.

    Args:
        selections: ``(rating_key, reason)`` pairs the curator produced; picks
            without a reason are skipped (there is nothing to reuse or mine).
        prompt: The request the run answered ("" for the seed-track flow).
        playlist_title: Title the narrative step gave the playlist.
        result_id: The ``results`` row's id in the library cache ("" when the
            run did not save one); kept as a string link, not a foreign key —
            the two DBs stay independent files.
        source: ``"prompt"`` or ``"seed"``, matching ``results.type`` spelling.

    Returns:
        Selection rows stored (0 when nothing carried a reason).
    """
    rows = [
        (key.strip(), reason.strip()[:MAX_REASON_CHARS])
        for key, reason in selections
        if key.strip() and reason.strip()
    ]
    if not rows:
        return 0

    run_id = secrets.token_hex(8)
    created_at = datetime.now().isoformat(timespec="seconds")
    conn = ensure_db_initialized()
    try:
        conn.execute(
            "INSERT INTO runs (run_id, source, prompt, playlist_title, result_id, track_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, source, (prompt or "").strip(), (playlist_title or "").strip(), result_id or "", len(rows), created_at),
        )
        conn.executemany(
            "INSERT INTO selections (run_id, rating_key, reason, created_at) VALUES (?, ?, ?, ?)",
            [(run_id, key, reason, created_at) for key, reason in rows],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO tags (rating_key, tag, created_at) VALUES (?, ?, ?)",
            [
                (key, tag, created_at)
                for key, reason in rows
                for tag in _mine_tags(reason)
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def get_previous_reasons(rating_keys: list[str]) -> dict[str, str]:
    """Latest stored reason per track key ({} when the memory has nothing).

    Newest wins per key: a track curated again for a different request replaces
    the argument replayed for it.
    """
    conn = ensure_db_initialized()
    reasons: dict[str, str] = {}
    try:
        keys = [key for key in dict.fromkeys(rating_keys) if key]
        for start in range(0, len(keys), QUERY_PARAM_CHUNK):
            chunk = keys[start : start + QUERY_PARAM_CHUNK]
            params: list[Any] = chunk
            rows = conn.execute(
                "SELECT rating_key, created_at, reason FROM selections "
                "WHERE rating_key IN ({}) ORDER BY created_at".format(", ".join(["?"] * len(chunk))),
                params,
            ).fetchall()
            for row in rows:
                reasons[row["rating_key"]] = row["reason"]
    except sqlite3.Error as e:
        logger.warning("Curation memory unreadable, replaying no reasons: %s", e)
        return {}
    finally:
        conn.close()
    return reasons


def get_reason_tags(rating_keys: list[str]) -> dict[str, set[str]]:
    """Mined tags per track key ({} when the memory has nothing)."""
    conn = ensure_db_initialized()
    found: dict[str, set[str]] = {}
    try:
        keys = [key for key in dict.fromkeys(rating_keys) if key]
        for start in range(0, len(keys), QUERY_PARAM_CHUNK):
            chunk = keys[start : start + QUERY_PARAM_CHUNK]
            rows = conn.execute(
                "SELECT rating_key, tag FROM tags WHERE rating_key IN ({})".format(", ".join(["?"] * len(chunk))),
                chunk,
            ).fetchall()
            for row in rows:
                found.setdefault(row["rating_key"], set()).add(row["tag"])
    except sqlite3.Error as e:
        logger.warning("Curation tags unreadable: %s", e)
        return {}
    finally:
        conn.close()
    return found


def keys_with_tags(tags: list[str], match: str = "any") -> set[str]:
    """Track keys whose stored reasons were mined into the *tags*.

    ``match="all"`` keeps only tracks every tag of which the agent has said
    yes; the default answers "curated at least once in this language".
    """
    wanted = sorted({tag for tag in (_clean_tag(t) for t in tags) if tag})
    if not wanted:
        return set()
    conn = ensure_db_initialized()
    try:
        params: list[Any] = wanted
        clause = ", ".join(["?"] * len(wanted))
        if match == "all":
            rows = conn.execute(
                f"SELECT rating_key FROM tags WHERE tag IN ({clause}) "
                "GROUP BY rating_key HAVING COUNT(DISTINCT tag) = ?",
                params + [len(wanted)],
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT DISTINCT rating_key FROM tags WHERE tag IN ({clause})",
                params,
            ).fetchall()
    except sqlite3.Error as e:
        logger.warning("Curation tags unreadable: %s", e)
        return set()
    finally:
        conn.close()
    return {row["rating_key"] for row in rows}


def has_reasoning_data() -> bool:
    """Whether any selection reason has been stored (gates replay + affinity).

    The EXISTS subquery form is deliberate: the scalar ``EXISTS(selections)``
    spelling is a SQLite syntax error, and every ``sqlite3.Error`` below reports
    "no memory", so the wrong form would silently switch the feature off.
    """
    conn = ensure_db_initialized()
    try:
        return conn.execute("SELECT EXISTS (SELECT 1 FROM selections)").fetchone()[0] == 1
    except sqlite3.Error as e:
        logger.warning("Curation memory unreadable, treating as empty: %s", e)
        return False
    finally:
        conn.close()


def reasoning_status() -> dict:
    """Curation memory state: stored runs, reasons and the mined tag vocabulary.

    The ``tags`` list is the vocabulary an agent has actually curated with, so a
    UI can offer it where the filter chips are without this module knowing any
    dataset specifics; counts are stored tag rows, so a tag reused across runs
    ranks ahead of one the agent used once.
    """
    conn = ensure_db_initialized()
    payload: dict[str, Any] = {
        "runs": 0,
        "selections": 0,
        "tracks_with_reasons": 0,
        "tagged_tracks": 0,
        "tags": [],
        "last_run_at": None,
    }
    try:
        payload["runs"] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        payload["selections"] = conn.execute("SELECT COUNT(*) FROM selections").fetchone()[0]
        payload["tracks_with_reasons"] = conn.execute(
            "SELECT COUNT(DISTINCT rating_key) FROM selections"
        ).fetchone()[0]
        payload["tagged_tracks"] = conn.execute(
            "SELECT COUNT(DISTINCT rating_key) FROM tags"
        ).fetchone()[0]
        payload["tags"] = [
            {"tag": row[0], "count": row[1]}
            for row in conn.execute(
                "SELECT tag, COUNT(*) FROM tags GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT ?",
                (MAX_STATUS_TAGS,),
            ).fetchall()
        ]
        payload["last_run_at"] = conn.execute("SELECT MAX(created_at) FROM runs").fetchone()[0]
        payload["error"] = None
    except sqlite3.Error as e:
        logger.warning("Curation memory status unreadable: %s", e)
        payload["error"] = str(e)
    finally:
        conn.close()
    return payload
