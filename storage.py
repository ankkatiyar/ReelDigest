"""
storage.py  —  ReelDigest summary archive

Persists every finished job to two places:
  * SQLite (reeldigest.db)   — full record incl. transcript and OCR text,
                               the source of truth for referring back later.
  * CSV    (reel_summaries.csv) — a lighter, always-live spreadsheet view
                                  (Excel opens it directly).

Also serves the archive back: recent() for history, search() for asking
"do I have anything about X?".  search() combines two retrievers —
FTS5 keyword matching over summary/transcript/OCR (catches exact names and
phrases) and cosine similarity over Ollama embeddings of the summary
(catches paraphrase).

Every write is best-effort: a storage failure must never fail a job, so
persist() swallows and logs its own errors.  Embedding is best-effort too —
a row with no embedding stays keyword-searchable.
"""

import contextlib
import csv
import logging
import os
import re
import sqlite3
import threading

import numpy as np
import requests

log = logging.getLogger(__name__)

_db_path: str = ""
_csv_path: str = ""
_lock = threading.Lock()

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Relevance floors, measured against the real archive rather than guessed.
# Real questions that have a match scored >= 0.685; questions with no match in
# the archive, and small talk, scored <= 0.578. Keyword side: exact tool-name
# queries scored >= 0.682, while small talk that merely shares a word with a
# video topped out at 0.557. The floors sit in those gaps.
SEMANTIC_FLOOR = float(os.getenv("SEARCH_SEMANTIC_FLOOR", "0.62"))
KEYWORD_FLOOR  = float(os.getenv("SEARCH_KEYWORD_FLOOR",  "0.60"))
KEYWORD_BONUS  = 0.15   # ranking nudge so exact-term hits sort above pure paraphrase

# Dropped from keyword queries: too common to narrow anything down, and they
# are what small talk ("how are you") is mostly made of.
_STOPWORDS = {
    "a", "about", "am", "an", "and", "any", "anything", "are", "as", "at", "be",
    "can", "could", "did", "do", "does", "for", "from", "get", "give", "got",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "know", "like",
    "me", "my", "of", "on", "or", "please", "que", "show", "so", "some",
    "something", "tell", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your",
}

# CSV omits transcript/ocr_text on purpose — those are large and only useful
# in the DB. Summary is multi-line but the csv module quotes it correctly.
_CSV_COLUMNS = [
    "created_at", "finished_at", "status", "url",
    "elapsed_s", "retries_used", "summary", "error",
]


@contextlib.contextmanager
def _connect():
    """Open the archive, commit on success, and always close the handle.

    `with sqlite3.connect(...)` alone commits but leaves the file open, which
    on Windows keeps a lock on the database file.
    """
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init(db_path: str, csv_path: str) -> None:
    """Create the DB table and CSV header if they don't exist yet."""
    global _db_path, _csv_path
    _db_path, _csv_path = db_path, csv_path
    try:
        with _connect() as conn:
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
                    chat_id      INTEGER,
                    embedding    BLOB
                )
                """
            )
            columns = {r[1] for r in conn.execute("PRAGMA table_info(summaries)")}
            if "chat_id" not in columns:
                conn.execute("ALTER TABLE summaries ADD COLUMN chat_id INTEGER")
            if "embedding" not in columns:
                conn.execute("ALTER TABLE summaries ADD COLUMN embedding BLOB")

            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
                    job_id UNINDEXED, summary, transcript, ocr_text
                )
                """
            )
            # Backfill the keyword index for rows archived before it existed.
            conn.execute(
                """
                INSERT INTO summaries_fts (job_id, summary, transcript, ocr_text)
                SELECT job_id, COALESCE(summary, ''), COALESCE(transcript, ''),
                       COALESCE(ocr_text, '')
                FROM summaries
                WHERE status = 'done'
                  AND job_id NOT IN (SELECT job_id FROM summaries_fts)
                """
            )
        if not os.path.exists(_csv_path) or os.path.getsize(_csv_path) == 0:
            with open(_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)
        log.info("Summary archive ready: %s + %s", _db_path, _csv_path)
    except Exception as exc:
        log.exception("Storage init failed (summaries will not be saved): %s", exc)


def embed(text: str, kind: str = "document"):
    """Return an L2-normalised float32 vector for text, or None if unavailable.

    nomic-embed-text is trained with task prefixes and scores poorly without
    them — stored summaries are documents, a user's question is a query.

    Best-effort by design: Ollama may be down or the embedding model may not
    be pulled, and neither is worth failing a job over.
    """
    text = (text or "").strip()
    if not text:
        return None
    if "nomic" in _EMBED_MODEL:
        text = f"search_{'query' if kind == 'query' else 'document'}: {text}"
    try:
        resp = requests.post(
            f"{_OLLAMA_HOST}/api/embeddings",
            json={"model": _EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if not norm:
            return None
        return vec / norm
    except Exception as exc:
        log.warning("Embedding failed (%s) - row stays keyword-searchable.", exc)
        return None


def persist(job) -> None:
    """Write one finished job to SQLite and append it to the CSV.

    Duck-typed on the Job dataclass — reads the transcript and OCR text from
    the internal _transcript / _ocr_text attributes set by the worker.
    Called once per job from the worker thread; never raises.
    """
    if not _db_path:
        return

    # Embed before taking the lock — it is a network call to Ollama.
    vector = embed(job.summary) if job.status == "done" else None

    with _lock:
        try:
            with _connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO summaries (
                        job_id, url, status, summary, transcript, ocr_text,
                        error, elapsed_s, retries_used,
                        created_at, started_at, finished_at, chat_id, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id, job.url, job.status, job.summary,
                        getattr(job, "_transcript", None),
                        getattr(job, "_ocr_text", None),
                        job.error, job.elapsed_s, job.retries_used,
                        job.created_at, job.started_at, job.finished_at,
                        getattr(job, "_chat_id", None),
                        None if vector is None else vector.tobytes(),
                    ),
                )
                if job.status == "done":
                    conn.execute(
                        "DELETE FROM summaries_fts WHERE job_id = ?", (job.job_id,)
                    )
                    conn.execute(
                        """
                        INSERT INTO summaries_fts (job_id, summary, transcript, ocr_text)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            job.job_id, job.summary or "",
                            getattr(job, "_transcript", None) or "",
                            getattr(job, "_ocr_text", None) or "",
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


def backfill_embeddings() -> int:
    """Embed archived summaries that have no vector yet. Returns the count done.

    Runs on a background thread at startup, so rows saved before embedding
    existed (or while Ollama was down) become searchable without a manual step.
    """
    if not _db_path:
        return 0
    try:
        with _connect() as conn:
            pending = conn.execute(
                """
                SELECT job_id, summary FROM summaries
                WHERE status = 'done' AND embedding IS NULL
                  AND summary IS NOT NULL AND summary != ''
                """
            ).fetchall()
    except Exception as exc:
        log.exception("Could not list rows needing embeddings: %s", exc)
        return 0

    done = 0
    for job_id, summary in pending:
        vector = embed(summary)
        if vector is None:
            log.warning("Embedding backfill stopped after %d rows.", done)
            break
        with _lock:
            try:
                with _connect() as conn:
                    conn.execute(
                        "UPDATE summaries SET embedding = ? WHERE job_id = ?",
                        (vector.tobytes(), job_id),
                    )
                done += 1
            except Exception as exc:
                log.exception("Failed to store embedding for %s: %s", job_id, exc)

    if done:
        log.info("Embedded %d archived summaries.", done)
    return done


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression, or "" if nothing useful.

    Each term is quoted, so punctuation and FTS5 operators in the user's
    question cannot change the query's meaning.
    """
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    terms = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    return " OR ".join(f'"{t}"' for t in terms)


def search(chat_id: int, query: str, limit: int = 5) -> list[dict]:
    """Find archived summaries relevant to a free-text question.

    Runs both retrievers and merges them: cosine similarity over summary
    embeddings catches paraphrase, FTS5 keyword matching catches exact names
    the embedding may gloss over. Results below the relevance floors are
    dropped, so small talk returns nothing rather than a bad guess.

    Returns dicts with url, summary, finished_at, score and matched_by.
    Never raises.
    """
    if not _db_path or not (query or "").strip():
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, url, summary, elapsed_s, finished_at, embedding
                FROM summaries
                WHERE status = 'done' AND (chat_id = ? OR chat_id IS NULL)
                """,
                (chat_id,),
            ).fetchall()

            match = _fts_query(query)
            keyword_hits = set()
            if match:
                keyword_hits = {
                    r["job_id"]
                    for r in conn.execute(
                        """
                        SELECT f.job_id FROM summaries_fts f
                        JOIN summaries s ON s.job_id = f.job_id
                        WHERE summaries_fts MATCH ?
                          AND s.status = 'done'
                          AND (s.chat_id = ? OR s.chat_id IS NULL)
                        ORDER BY bm25(summaries_fts)
                        LIMIT ?
                        """,
                        (match, chat_id, limit * 4),
                    ).fetchall()
                }

        query_vec = embed(query, kind="query")
        # No query vector means Ollama is unreachable. Rather than return
        # nothing, fall back to keyword-only matching for every row.
        degraded = query_vec is None

        results = []
        for row in rows:
            cosine = 0.0
            if query_vec is not None and row["embedding"]:
                doc = np.frombuffer(row["embedding"], dtype=np.float32)
                cosine = float(np.dot(query_vec, doc))

            is_keyword = row["job_id"] in keyword_hits
            # A keyword hit still has to be topically plausible, or small talk
            # that happens to share a word with one video would match it.
            if degraded or not row["embedding"]:
                if not is_keyword:
                    continue
                matched_by = "keyword"
            elif cosine >= SEMANTIC_FLOOR:
                matched_by = "keyword+semantic" if is_keyword else "semantic"
            elif is_keyword and cosine >= KEYWORD_FLOOR:
                matched_by = "keyword"
            else:
                continue

            results.append({
                "url": row["url"],
                "summary": row["summary"],
                "finished_at": row["finished_at"],
                "elapsed_s": row["elapsed_s"],
                "score": round(cosine + (KEYWORD_BONUS if is_keyword else 0.0), 4),
                "matched_by": matched_by,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]
    except Exception as exc:
        log.exception("Search failed for %r: %s", query, exc)
        return []


def recent(chat_id: int, limit: int = 5) -> list[dict]:
    """Return that chat's most recent completed summaries, newest first.

    Rows with a NULL chat_id predate per-chat attribution, so they are
    included too rather than being stranded. Never raises.
    """
    if not _db_path:
        return []
    try:
        with _connect() as conn:
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
