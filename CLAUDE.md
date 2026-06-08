# ReelDigest — Claude Code instructions

## Project Overview

ReelDigest is a fully-local Instagram Reel and carousel summariser. Downloads
posts (including image-only carousels), transcribes audio (Whisper), reads
on-screen text (EasyOCR), and summarises with a local LLM (Ollama). Exposed
as a FastAPI REST API and a Telegram bot. Runs entirely on the user's machine
— no cloud APIs.

For full architecture and usage details, see `README.md`. This file is
specifically for working preferences, conventions, and Git rules.

## Architecture Map

- **`reel_summarizer.py`** — core pipeline: download → transcribe (Whisper) → OCR
  (EasyOCR) → summarise (Ollama). Two-stage download: yt-dlp first, Instagram
  API fallback for image-only carousels.
- **`server.py`** — FastAPI server, async job queue, worker thread. Loads
  models once (`_load_models`). Hardcoded `device="cpu"` due to CUDA/cuDNN
  mismatch.
- **`bot.py`** — Telegram bot. Starts via FastAPI lifespan in the same process
  as the server. Commands: /start, /last, /status, /history.
- **`start.ps1`** — launcher: loads `.env` and runs `server.py` with venv Python.
- **`tests/`** — unittest-based test suite.

## Working Preferences

- **Plan before touching.** For anything non-trivial, propose a plan and wait
  for approval before editing.
- **Read first.** Always read relevant files before writing. Match the
  existing code style.
- **Small, reviewable edits.** Don't rewrite whole files. Don't clean up
  surrounding code unless asked.
- **Don't add dependencies without flagging them first.** This project keeps
  dependencies minimal — every new package needs a real justification.
- When uncertain, ask. Don't guess on architecture or priorities.
- Default to no comments. Only add one when the WHY is non-obvious — a hidden
  constraint, a workaround, a subtle invariant.

## Rules and Conventions

- **Never commit `.env` or any API key.** `.gitignore` covers it but verify
  before every commit.
- **Never commit `instagram_cookies.txt` or any `*_cookies.txt` file.** These
  contain real auth tokens.
- Runtime artifacts (`reel_summaries*.txt`, `*.mp4`, `reels_*/`) stay
  gitignored. Don't add them to tracked files.
- Always use the venv Python (`.\venv\Scripts\python.exe`), not the system
  Python. The system Python is missing required packages.
- `device="cpu"` is hardcoded in `server.py:_load_models` for a reason
  (CUDA/cuDNN mismatch). Don't change to "cuda" without resolving the
  underlying driver issue.
- `OLLAMA_NUM_GPU=0` in `.env` is required on this hardware (Vulkan OOM on
  the RTX 4060). Don't remove unless GPU inference is verified working.

## Git Workflow

1. After every completed task, run `git status` and propose a commit plan
   showing which files go in which commit, with clear messages.
2. **Wait for explicit approval before committing.**
3. Group related changes logically — not one mega-commit unless the work is
   genuinely a single concern.
4. Commit messages: describe what changed and why. "Add cookies-based auth for
   Instagram downloads" not "updates". Brief but specific.
5. Flag uncertainty in commit messages: "WIP: X — runs but not yet tested".
6. After commits are approved, ask whether to push to GitHub.
7. **Do NOT add `Co-Authored-By: Claude` trailers, `🤖 Generated with Claude
   Code` footers, or any other AI-tool attribution to commit messages.**
   Commit messages should describe what changed and why, with no AI-tool
   metadata.
8. **Never `git push --force`, `git reset --hard`, or `git rebase`** without
   explicit instruction in the current session.

## Quirks Worth Knowing

- **yt-dlp image carousel bug.** yt-dlp raises "No video formats found!" for
  image-only carousel posts. Handled by Stage 2 fallback in
  `_fetch_instagram_images` calling Instagram's private API directly. Don't
  remove the fallback even if yt-dlp seems to work for a particular URL.
- **Instagram shortcode → media ID conversion** uses Instagram's custom base-64
  alphabet (`A-Za-z0-9-_`). Not standard base64. Documented in
  `reel_summarizer.py`.
- **Cookies expire periodically** (weeks–months). User must re-export from the
  browser extension. If downloads start failing with auth errors, this is the
  first thing to check.
- **CUDA/cuDNN/Vulkan are a known mess on this machine.** Multiple workarounds
  exist (`device="cpu"`, `OLLAMA_NUM_GPU=0`). Don't try to "fix" individual
  pieces — the whole GPU stack needs resolution together (Phase 4).

## Current Priority Stack

| Status | Item |
|---|---|
| ✅ Done | Phase 1: FastAPI server + async job queue |
| ✅ Done | Phase 2: Telegram bot with push notifications |
| ✅ Done | Phase 3a: Instagram cookies auth + image carousel support |
| ⏳ Pending | Phase 3b: Tailscale setup (phone access from any network) |
| ⏳ Pending | Phase 4: Fix GPU inference (cuDNN / Vulkan issues) |
| 🅿️ Parked | Phase 5: Profile analyser — crawl all posts on a profile, meta-summary |