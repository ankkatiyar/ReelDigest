# ReelDigest

A fully local Instagram Reel and carousel post summariser. Downloads posts
(including image-only carousels), transcribes audio (Whisper), reads on-screen
text (EasyOCR), and summarises with a local LLM (Ollama). Works for any topic —
medical, fitness, finance, tech, cooking, etc. Exposed as a REST API and a
Telegram bot. Runs entirely on your own machine.

## How to run

```powershell
# First time only: install dependencies
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Every time: start the server (loads .env automatically)
.\start.ps1
```

The server starts at `http://0.0.0.0:8000`. The Telegram bot starts
automatically if `TELEGRAM_TOKEN` is set in `.env`.

## Key files

| File | Purpose |
|---|---|
| `reel_summarizer.py` | Core pipeline: download → transcribe → OCR → summarise |
| `server.py` | FastAPI server + async job queue + worker thread |
| `bot.py` | Telegram bot (same process as server, starts via lifespan) |
| `start.ps1` | Launcher: loads `.env` and runs server with venv Python |
| `.env.example` | All config options documented with defaults |
| `instagram_cookies.txt` | Your Instagram session (see setup below — gitignored) |
| `urls.txt` | Input for the CLI mode (one URL per line) |

## Instagram authentication (required)

Meta disabled anonymous API access. Every download now requires a valid
Instagram session exported as a Netscape cookies file.

**One-time setup:**

1. Install the **"Get cookies.txt LOCALLY"** extension in Chrome or Edge
2. Log in to `instagram.com` in that browser
3. Click the extension icon, select `instagram.com`, click **Export**
4. Save the file as `instagram_cookies.txt` in the project root
5. Add to `.env`: `INSTAGRAM_COOKIES_FILE=instagram_cookies.txt`

The cookies file is gitignored. Re-export when Instagram invalidates your
session (usually weeks–months).

## Environment / config

Copy `.env.example` to `.env` and fill in:
- `INSTAGRAM_COOKIES_FILE=instagram_cookies.txt` — required for all downloads
- `TELEGRAM_TOKEN` — from @BotFather
- `TELEGRAM_ALLOWED_USERS` — your Telegram user ID (get from @userinfobot)
- `OLLAMA_NUM_GPU=0` — required on this machine (Vulkan OOM on GPU)

## Download pipeline (two-stage)

**Stage 1 — yt-dlp** handles video reels and video carousels (`format: best`).

**Stage 2 — Instagram API fallback** (`_fetch_instagram_images`) kicks in when
yt-dlp raises "No video formats found!" — a known yt-dlp extractor limitation
for image-only carousel posts. It calls
`instagram.com/api/v1/media/{media_id}/info/` directly with the session cookies
to get slide image URLs and downloads them as `.jpg` files.

The shortcode → numeric media ID conversion uses Instagram's base-64 alphabet:
`ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_`

## API

```
POST /summarize   { "url": "https://instagram.com/reel/..." }  → 202 { job_id }
GET  /jobs/{id}   → { status, current_step, summary, elapsed_s, ... }
GET  /jobs        → list all jobs (filter: ?status=done|failed|pending|processing)
GET  /health      → server + model readiness
```

Supported URL patterns:
- `instagram.com/reel/{shortcode}/` — video reel
- `instagram.com/p/{shortcode}/` — photo post or image/video carousel
- `instagram.com/p/{shortcode}/?img_index=N` — deep-link to a specific slide (whole carousel is still processed)

## Telegram bot commands

| Command | Description |
|---|---|
| `/start` | Welcome message and usage |
| `/last` | Live status of your most recent job |
| `/status` | Server health, model status, queue depth |
| `/history` | Last 5 completed summaries |

Send any Instagram URL directly to the bot to queue a job. You get a push
notification when it finishes.

## CLI mode (bypasses the server)

```powershell
.\venv\Scripts\python.exe reel_summarizer.py -i urls.txt -o reel_summaries.txt --device cpu --ollama-num-gpu 0
```

## Run tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known issues / workarounds

- **CUDA/cuDNN mismatch**: EasyOCR's bundled PyTorch doesn't match the installed
  cuDNN. Workaround: `--device cpu` in CLI, hardcoded `device="cpu"` in
  `server.py:_load_models`.
- **Ollama Vulkan OOM**: Ollama tries to use the GPU via Vulkan and runs out of
  VRAM. Workaround: `OLLAMA_NUM_GPU=0` in `.env` forces CPU inference.
- **Always use the venv Python**: `python server.py` uses system Python (missing
  packages). Use `.\start.ps1` or `.\venv\Scripts\python.exe server.py`.
- **yt-dlp image carousel bug**: yt-dlp raises "No video formats found!" for
  image-only carousel posts — handled automatically by Stage 2 fallback.

## Hardware

Currently running on **MSI Katana 15 B13 V** (i7-13620H, RTX 4060 8 GB,
16 GB DDR5). Everything runs on CPU due to the CUDA/Vulkan issues above. Once
the driver/library mismatch is resolved, GPU inference will significantly speed
up Whisper and OCR.

## Adding a new Telegram bot command

1. Add a handler function `async def _cmd_name(update, ctx)` in `bot.py`
2. Register it in `start()`: `_app.add_handler(CommandHandler("name", _cmd_name))`
3. Restart the server

## Phase roadmap

- [x] Phase 1: FastAPI server + async job queue
- [x] Phase 2: Telegram bot with push notifications
- [x] Phase 3a: Instagram authentication (cookies) + image carousel support
- [ ] Phase 3b: Tailscale setup (reach server from phone on any network)
- [ ] Phase 4: Fix GPU inference (cuDNN / Vulkan issues on MSI Katana)
- [ ] Phase 5: Profile analyser — given an Instagram (or YouTube) profile URL,
      crawl all posts, run the summarise pipeline on each, then produce a
      meta-analysis via LLM covering: content evolution timeline, engagement
      pivot point, originality/novelty ranking per post (1–5 scale),
      monetisation potential rating, and AI-replication feasibility assessment.
      Planned as a standalone CLI script first, then optionally a Telegram
      command (`/profile <username>`).
