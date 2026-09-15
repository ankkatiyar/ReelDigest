"""
storage.py  —  ReelDigest summary archive

Persists every finished job to two places:
  * SQLite (reeldigest.db)   — full record incl. transcript and OCR text,
                               the source of truth for referring back later.
  * CSV    (reel_summaries.csv) — a lighter, always-live spreadsheet view
                                  (Excel opens it directly).

Both writes are best-effort: a storage failure must never fail a job, so
persist() swallows and logs its own errors.
"""

import csv
import logging
import os
import sqlite3
import threading

log = logging.getLogger(__name__)

_db_path: str = ""
_csv_path: str = ""
_lock = threading.Lock()

# CSV omits transcript/ocr_text on purpose — those are large and only useful
# in the DB. Summary is multi-line but the csv module quotes it correctly.
_CSV_COLUMNS = [
    "created_at", "finished_at", "status", "url",
    "elapsed_s", "retries_used", "summary", "error",
]


def init(db_path: str, csv_path: str) -> None:
    """Create the DB table and CSV header if they don't exist yet."""
    global _db_path, _csv_path
    _db_path, _csv_path = db_path, csv_path
    try:
        with sqlite3.connect(_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    job_id       TEXT PRIMARY KEY,
                    url          TEXT,
                    status       TEXT,
                    summary      TEXT,
                    transcript   TEXT,
                    ocr_text     TEXT,
                    error        TEXT,
                    elapsed_s    REAL,
                    retries_used INTEGER,
                    created_at   TEXT,
                    started_at   TEXT,
                    finished_at  TEXT,
                    chat_id      INTEGER
                )
                """
            )
            columns = {r[1] for r in conn.execute("PRAGMA table_info(summaries)")}
            if "chat_id" not in columns:
                conn.execute("ALTER TABLE summaries ADD COLUMN chat_id INTEGER")
        if not os.path.exists(_csv_path) or os.path.getsize(_csv_path) == 0:
            with open(_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)
        log.info("Summary archive ready: %s + %s", _db_path, _csv_path)
    except Exception as exc:
        log.exception("Storage init failed (summaries will not be saved): %s", exc)


def persist(job) -> None:
    """Write one finished job to SQLite and append it to the CSV.

    Duck-typed on the Job dataclass — reads the transcript and OCR text from
    the internal _transcript / _ocr_text attributes set by the worker.
    Called once per job from the worker thread; never raises.
    """
    if not _db_path:
        return
    with _lock:
        try:
            with sqlite3.connect(_db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO summaries (
                        job_id, url, status, summary, transcript, ocr_text,
                        error, elapsed_s, retries_used,
                        created_at, started_at, finished_at, chat_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id, job.url, job.status, job.summary,
                        getattr(job, "_transcript", None),
                        getattr(job, "_ocr_text", None),
                        job.error, job.elapsed_s, job.retries_used,
                        job.created_at, job.started_at, job.finished_at,
                        getattr(job, "_chat_id", None),
                    ),
                )
        except Exception as exc:
            log.exception("Failed to write job %s to DB: %s", job.job_id, exc)

        try:
            with open(_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    job.created_at, job.finished_at, job.status, job.url,
                    job.elapsed_s, job.retries_used, job.summary or "", job.error or "",
                ])
        except Exception as exc:
            log.exception("Failed to append job %s to CSV: %s", job.job_id, exc)


def recent(chat_id: int, limit: int = 5) -> list[dict]:
    """Return that chat's most recent completed summaries, newest first.

    Rows with a NULL chat_id predate per-chat attribution, so they are
    included too rather than being stranded. Never raises.
    """
    if not _db_path:
        return []
    try:
        with sqlite3.connect(_db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT url, summary, elapsed_s, finished_at
                FROM summaries
                WHERE status = 'done'
                  AND (chat_id = ? OR chat_id IS NULL)
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.exception("Failed to read recent summaries: %s", exc)
        return []
