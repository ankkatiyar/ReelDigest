"""
bot.py  - ReelDigest Telegram bot

Runs inside the same process as server.py (started from its lifespan).
Users send an Instagram Reel or carousel URL; the bot queues the job and
sends the summary back when it is done.

Required env var:
    TELEGRAM_TOKEN          - bot token from @BotFather

Optional env var:
    TELEGRAM_ALLOWED_USERS  - comma-separated Telegram user IDs allowed to
                               use the bot (e.g. "123456789,987654321").
                               Leave unset to allow anyone who finds the bot.
"""

import asyncio
import html
import logging
import os
import re
from collections import defaultdict
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ALLOWED: set[int] = set(
    int(x.strip())
    for x in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if x.strip().lstrip("-").isdigit()
)

_INSTAGRAM_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:reel|p)/[A-Za-z0-9_\-]+/?[^\s]*"
)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_app: Optional[Application] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# Maps chat_id -> list of job_ids submitted by that user (newest last)
_user_jobs: dict[int, list[str]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Public API  (called from server.py lifespan)
# ---------------------------------------------------------------------------

async def start(token: str) -> None:
    """Initialise the bot and start long-polling."""
    global _app, _main_loop
    _main_loop = asyncio.get_running_loop()

    _app = Application.builder().token(token).build()
    _app.add_handler(CommandHandler("start",   _cmd_start))
    _app.add_handler(CommandHandler("status",  _cmd_status))
    _app.add_handler(CommandHandler("last",    _cmd_last))
    _app.add_handler(CommandHandler("history", _cmd_history))
    _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))

    await _app.initialize()
    await _app.start()
    await _app.updater.start_polling(drop_pending_updates=True)
    log.info("Telegram bot started and polling.")


async def stop() -> None:
    """Gracefully shut the bot down."""
    if _app:
        await _app.updater.stop()
        await _app.stop()
        await _app.shutdown()
        log.info("Telegram bot stopped.")


# ---------------------------------------------------------------------------
# Permission check
# ---------------------------------------------------------------------------

def _is_allowed(update: Update) -> bool:
    if not _ALLOWED:
        return True
    return update.effective_user.id in _ALLOWED


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    await update.message.reply_text(
        "<b>ReelDigest Bot</b>\n\n"
        "Send me any public Instagram Reel or carousel link and I'll send "
        "back a bullet-point summary, fully processed on your local machine.\n\n"
        "Just paste the URL and I'll handle the rest.\n\n"
        "<b>Commands</b>\n"
        "/last    - check the status of your last job\n"
        "/status  - server health &amp; queue depth\n"
        "/history - your last 5 completed summaries",
        parse_mode=ParseMode.HTML,
    )


async def _cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return

    from server import (
        _models_ready, _job_queue, _active_job_id,
        _jobs, _jobs_lock, WHISPER_MODEL, OLLAMA_MODEL,
    )

    ready = _models_ready.is_set()
    with _jobs_lock:
        total  = len(_jobs)
        done   = sum(1 for j in _jobs.values() if j.status == "done")
        failed = sum(1 for j in _jobs.values() if j.status == "failed")
        queued = sum(1 for j in _jobs.values() if j.status == "pending")

    lines = [
        f"{'✅' if ready else '⏳'} Models: <b>{'ready' if ready else 'loading...'}</b>",
        f"🧠 Whisper: <code>{WHISPER_MODEL}</code>  |  Ollama: <code>{OLLAMA_MODEL}</code>",
        f"📋 Queue: <b>{queued}</b> waiting",
        f"⚙️ Active job: <code>{_active_job_id or 'none'}</code>",
        f"📊 All-time: {done} done, {failed} failed, {total} total",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def _cmd_last(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the status (and summary if done) of the user's most recent job."""
    if not _is_allowed(update):
        return

    chat_id = update.effective_chat.id
    job_ids = _user_jobs.get(chat_id, [])

    if not job_ids:
        await update.message.reply_text(
            "You haven't submitted any jobs yet.\nSend me an Instagram URL to get started."
        )
        return

    from server import _jobs, _jobs_lock

    # Walk backwards to find the most recent job still in memory
    job = None
    with _jobs_lock:
        for jid in reversed(job_ids):
            if jid in _jobs:
                job = _jobs[jid]
                break

    if not job:
        await update.message.reply_text(
            "Your last job is no longer in memory (server was restarted).\n"
            "Send a new URL to start fresh."
        )
        return

    step_map = {
        "downloading":  "📥 Downloading...",
        "transcribing": "🎙 Transcribing audio...",
        "ocr":          "🔍 Reading on-screen text...",
        "summarizing":  "🧠 Summarizing...",
    }

    if job.status == "pending":
        text = (
            f"⏳ <b>Still in queue</b>\n\n"
            f"Job: <code>{job.job_id}</code>\n"
            f"Queued at: {job.created_at}"
        )
    elif job.status == "processing":
        step_label = step_map.get(job.current_step or "", f"Step: {job.current_step}")
        text = (
            f"⚙️ <b>Processing now</b>\n\n"
            f"Job: <code>{job.job_id}</code>\n"
            f"{step_label}"
        )
    elif job.status == "done":
        text = (
            f"✅ <b>Summary</b>\n\n"
            f"{html.escape(job.summary or '')}\n\n"
            f"⏱ Done in {_fmt_elapsed(job.elapsed_s)}"
        )
    else:
        text = (
            f"❌ <b>Failed</b>\n\n"
            f"<code>{html.escape(job.error or 'unknown error')}</code>\n\n"
            "The reel may be private, deleted, or temporarily unavailable."
        )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def _cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return

    chat_id = update.effective_chat.id
    job_ids = _user_jobs.get(chat_id, [])

    if not job_ids:
        await update.message.reply_text("No jobs submitted yet.")
        return

    from server import _jobs, _jobs_lock

    with _jobs_lock:
        done_jobs = [
            _jobs[jid] for jid in reversed(job_ids)
            if jid in _jobs and _jobs[jid].status == "done"
        ][:5]

    if not done_jobs:
        await update.message.reply_text(
            "No completed summaries yet.\nUse /last to check if a job is still running."
        )
        return

    parts = []
    for j in done_jobs:
        short_url = j.url.split("?")[0].rstrip("/").split("/")[-1]
        parts.append(
            f"<b>{html.escape(short_url)}</b>  <i>({_fmt_elapsed(j.elapsed_s)})</i>\n"
            f"{html.escape(j.summary or '')}"
        )

    await update.message.reply_text(
        "\n\n---\n\n".join(parts),
        parse_mode=ParseMode.HTML,
    )


# ---------------------------------------------------------------------------
# Message handler  (the main flow)
# ---------------------------------------------------------------------------

async def _on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("You are not authorised to use this bot.")
        return

    text  = (update.message.text or "").strip()
    match = _INSTAGRAM_RE.search(text)

    if not match:
        await update.message.reply_text(
            "Please send a public Instagram Reel or carousel link.\n\n"
            "<i>Example:</i>\n"
            "<code>https://www.instagram.com/reel/XXXXX/</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    url     = match.group(0)
    chat_id = update.effective_chat.id

    from server import _submit_job, _models_ready, _models_error

    if not _models_ready.is_set():
        await update.message.reply_text("Models are still loading at startup — try again in a moment.")
        return
    if _models_error:
        await update.message.reply_text(f"Model loading failed:\n<code>{html.escape(_models_error)}</code>", parse_mode=ParseMode.HTML)
        return

    def _notify(job):
        if _app and _main_loop:
            asyncio.run_coroutine_threadsafe(
                _send_result(job, chat_id), _main_loop
            )

    job = _submit_job(url, notify_fn=_notify)

    # Track this job for the user so /last can find it
    _user_jobs[chat_id].append(job.job_id)

    queue_msg = (
        "you're next up"
        if job.queue_position == 0
        else f"{job.queue_position} job{'s' if job.queue_position > 1 else ''} ahead of you"
    )

    # html.escape the URL - the & in tracking params breaks Telegram's HTML parser
    await update.message.reply_text(
        f"⏳ <b>Got it!</b> Processing...\n\n"
        f"<code>{html.escape(url)}</code>\n\n"
        f"Queue: {queue_msg}\n"
        f"Use /last to check progress. I'll message you when it's ready.",
        parse_mode=ParseMode.HTML,
    )


# ---------------------------------------------------------------------------
# Result sender  (scheduled onto the asyncio loop from the worker thread)
# ---------------------------------------------------------------------------

async def _send_result(job, chat_id: int) -> None:
    if not _app:
        return

    if job.status == "done":
        text = (
            f"✅ <b>Summary</b>\n\n"
            f"{html.escape(job.summary or '')}\n\n"
            f"⏱ Done in {_fmt_elapsed(job.elapsed_s)}"
        )
    else:
        text = (
            f"❌ <b>Failed</b>\n\n"
            f"<code>{html.escape(job.error or 'unknown error')}</code>\n\n"
            "The post may be private, deleted, or temporarily unavailable.\n"
            "Try copying the link fresh from Instagram."
        )

    try:
        await _app.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        log.error("Failed to send result to chat %s: %s", chat_id, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"
