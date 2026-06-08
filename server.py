"""
server.py  —  ReelDigest async job API

Wraps the reel_summarizer pipeline behind a REST API with a background
job queue so the caller (Telegram bot, mobile shortcut, etc.) can submit
a URL and poll for results — or receive a push notification via a
registered completion callback.

Start:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /summarize          { "url": "https://instagram.com/reel/..." }
    GET  /jobs/{job_id}      current state of one job
    GET  /jobs               all jobs, newest first (optional ?status= filter)
    GET  /health             server + model readiness

Config (all via environment variables — see .env.example):
    WHISPER_MODEL, OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_NUM_GPU, ...
"""

import os
import queue
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import reel_summarizer as rs

# ---------------------------------------------------------------------------
# Config  (env-var driven, defaults tuned for weak CPU hardware)
# ---------------------------------------------------------------------------

WHISPER_MODEL  = os.getenv("WHISPER_MODEL",   "tiny")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",    "mistral")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST",     "http://localhost:11434")
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))   # 0 = CPU-only
OCR_LANG       = [s.strip() for s in os.getenv("OCR_LANG", "en").split(",")]
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "2.0"))
MAX_FRAMES     = int(os.getenv("MAX_FRAMES",     "60"))
DIFF_THRESHOLD = float(os.getenv("DIFF_THRESHOLD", "8.0"))
MAX_CHARS      = int(os.getenv("MAX_CHARS",      "6000"))
DL_ATTEMPTS    = int(os.getenv("DOWNLOAD_ATTEMPTS", "2"))
DL_TIMEOUT     = int(os.getenv("DOWNLOAD_TIMEOUT",  "30"))
RETRY_DELAY    = float(os.getenv("RETRY_DELAY",     "2.0"))
MAX_STORED_JOBS = int(os.getenv("MAX_STORED_JOBS", "200"))
INSTAGRAM_COOKIES = os.getenv("INSTAGRAM_COOKIES_FILE", "").strip() or None
PORT           = int(os.getenv("PORT", "8000"))
HOST           = os.getenv("HOST", "0.0.0.0")

# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id:      str
    url:         str
    status:      str        = "pending"    # pending | processing | done | failed
    current_step: Optional[str] = None     # downloading | transcribing | ocr | summarizing
    summary:     Optional[str] = None
    error:       Optional[str] = None
    elapsed_s:   Optional[float] = None
    retries_used: int        = 0
    queue_position: int      = 0           # updated on submit; rough estimate
    created_at:  str         = field(default_factory=lambda: _now())
    started_at:  Optional[str] = None
    finished_at: Optional[str] = None
    # Internal — never serialised to the API response
    _notify:     Optional[Callable] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_notify", None)
        return d


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_jobs: dict[str, Job]  = {}
_jobs_lock             = threading.Lock()
_job_queue: queue.Queue = queue.Queue()

_models_ready          = threading.Event()
_models_error: Optional[str] = None
_whisper_model         = None
_ocr_reader            = None
_active_job_id: Optional[str] = None
_temp_dir              = tempfile.mkdtemp(prefix="reeldigest_")

# Phase 2 hook: Telegram bot (or any module) calls register_completion_callback()
# to be notified when a job finishes.  Callbacks run from the worker thread.
_completion_callbacks: list[Callable[[Job], None]] = []


def register_completion_callback(fn: Callable[[Job], None]) -> None:
    """Register a function called whenever a job reaches 'done' or 'failed'.

    The function receives the finished Job and runs on the worker thread,
    so it must not block.  Hand async work off with
    asyncio.run_coroutine_threadsafe(coro, loop).
    """
    _completion_callbacks.append(fn)


# ---------------------------------------------------------------------------
# Model loading  (background thread — keeps startup fast)
# ---------------------------------------------------------------------------

def _load_models() -> None:
    global _whisper_model, _ocr_reader, _models_error
    import logging
    log = logging.getLogger(__name__)
    try:
        from faster_whisper import WhisperModel
        import easyocr
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        _ocr_reader    = easyocr.Reader(OCR_LANG, gpu=False, verbose=False)
        _models_ready.set()
        log.info("Models loaded successfully.")
    except Exception as exc:
        _models_error = str(exc)
        log.exception("Model loading failed: %s", exc)
        _models_ready.set()  # unblock health/worker so they can report the error


# ---------------------------------------------------------------------------
# Worker  (single thread — one reel at a time, CPU-friendly)
# ---------------------------------------------------------------------------

def _run_job(job: Job) -> None:
    """Execute the full pipeline for one job. Mutates job in place.
    Handles both single reels and multi-slide carousel posts."""
    global _active_job_id
    t0          = time.time()
    media_paths = []
    _active_job_id = job.job_id

    try:
        job.status     = "processing"
        job.started_at = _now()

        job.current_step = "downloading"
        media_paths, retries = rs.run_with_retries(
            lambda: rs.download_reel(
                job.url, _temp_dir,
                retries=3, socket_timeout=DL_TIMEOUT,
                cookies_file=INSTAGRAM_COOKIES,
            ),
            attempts=DL_ATTEMPTS,
            delay=RETRY_DELAY,
            label="download",
        )
        job.retries_used = retries

        all_transcripts, all_ocr = [], []

        for path in media_paths:
            job.current_step = "transcribing"
            if not rs.is_image(path):
                t = rs.transcribe(_whisper_model, path)
                if t:
                    all_transcripts.append(t)

            job.current_step = "ocr"
            o = rs.extract_media_text(
                path, _ocr_reader,
                FRAME_INTERVAL, MAX_FRAMES, DIFF_THRESHOLD,
            )
            if o:
                all_ocr.append(o)

        transcript = " ".join(all_transcripts)
        ocr_text   = "\n---\n".join(all_ocr)

        job.current_step = "summarizing"
        summary = rs.summarize(
            transcript, ocr_text,
            OLLAMA_MODEL, OLLAMA_HOST, MAX_CHARS, OLLAMA_NUM_GPU,
        )

        job.status       = "done"
        job.summary      = summary
        job.current_step = None
        job.elapsed_s    = round(time.time() - t0, 1)
        job.finished_at  = _now()

    except Exception as exc:
        job.status       = "failed"
        job.error        = rs.short_reason(exc)
        job.current_step = None
        job.elapsed_s    = round(time.time() - t0, 1)
        job.finished_at  = _now()

    finally:
        _active_job_id = None
        for path in media_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # Fire completion callbacks (Phase 2 Telegram bot plugs in here)
    for cb in _completion_callbacks:
        try:
            cb(job)
        except Exception:
            pass

    # Also fire any per-job notify (set by caller at submit time)
    if job._notify:
        try:
            job._notify(job)
        except Exception:
            pass


def _worker() -> None:
    _models_ready.wait()
    while True:
        job_id = _job_queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job:
            _run_job(job)
        _job_queue.task_done()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_models, daemon=True, name="model-loader").start()
    threading.Thread(target=_worker,      daemon=True, name="job-worker").start()

    _tg = None
    _telegram_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if _telegram_token:
        import bot as _tg
        await _tg.start(_telegram_token)

    yield

    if _tg:
        await _tg.stop()


app = FastAPI(
    title="ReelDigest API",
    description="Submit Instagram Reel URLs, get bullet-point summaries.",
    version="1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def must_be_instagram_reel(cls, v: str) -> str:
        v = v.strip()
        if "instagram.com/reel" not in v and "instagram.com/p/" not in v:
            raise ValueError("URL must be an Instagram Reel link (instagram.com/reel/…)")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Server and model readiness")
def health():
    loading = not _models_ready.is_set()
    failed  = _models_ready.is_set() and _models_error is not None
    ready   = _models_ready.is_set() and _models_error is None

    with _jobs_lock:
        counts = {"total": 0, "pending": 0, "processing": 0, "done": 0, "failed": 0}
        for j in _jobs.values():
            counts["total"] += 1
            counts[j.status] = counts.get(j.status, 0) + 1

    status_str = "ready" if ready else ("error" if failed else "loading")
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status":        status_str,
            "models_ready":  ready,
            "models_error":  _models_error,
            "whisper_model": WHISPER_MODEL,
            "ollama_model":  OLLAMA_MODEL,
            "ollama_host":   OLLAMA_HOST,
            "queue_depth":   _job_queue.qsize(),
            "active_job_id": _active_job_id,
            "jobs":          counts,
        },
    )


def _submit_job(url: str, notify_fn: Optional[Callable] = None) -> Job:
    """Create, store, and enqueue a job.

    Called by both the HTTP endpoint and the Telegram bot so neither
    needs to reach the other over HTTP.
    """
    job_id    = uuid.uuid4().hex[:12]
    queue_pos = _job_queue.qsize() + (1 if _active_job_id else 0)
    job       = Job(job_id=job_id, url=url, queue_position=queue_pos, _notify=notify_fn)
    with _jobs_lock:
        _evict_old_jobs()
        _jobs[job_id] = job
    _job_queue.put(job_id)
    return job


@app.post("/summarize", status_code=202, summary="Submit a reel URL for summarisation")
def submit(body: SummarizeRequest):
    if not _models_ready.is_set():
        raise HTTPException(status_code=503, detail="Models are still loading — wait a moment and try again.")
    if _models_error:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {_models_error}")
    job = _submit_job(body.url)
    return {
        "job_id":         job.job_id,
        "status":         "pending",
        "queue_position": job.queue_position,
        "poll_url":       f"/jobs/{job.job_id}",
        "created_at":     job.created_at,
    }


@app.get("/jobs/{job_id}", summary="Get the current state of one job")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict()


@app.get("/jobs", summary="List all jobs, newest first")
def list_jobs(status: Optional[str] = Query(default=None, description="Filter by status: pending | processing | done | failed")):
    with _jobs_lock:
        jobs = list(_jobs.values())

    jobs.sort(key=lambda j: j.created_at, reverse=True)

    if status:
        jobs = [j for j in jobs if j.status == status]

    return {"total": len(jobs), "jobs": [j.to_dict() for j in jobs]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _evict_old_jobs() -> None:
    """Remove oldest terminal jobs when the store exceeds MAX_STORED_JOBS.
    Must be called with _jobs_lock held."""
    terminal = [j for j in _jobs.values() if j.status in ("done", "failed")]
    if len(_jobs) >= MAX_STORED_JOBS and terminal:
        terminal.sort(key=lambda j: j.created_at)
        oldest = terminal[0]
        del _jobs[oldest.job_id]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
