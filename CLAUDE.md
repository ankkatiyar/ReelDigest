# ReelDigest

A fully local Instagram Reel and carousel post summariser. Downloads public posts, transcribes audio (Whisper), reads on-screen text (EasyOCR), and summarises with a local LLM (Ollama). Works for any topic — medical, fitness, finance, tech, cooking, etc. Exposed as a REST API and a Telegram bot — no cloud APIs, no login required.

## How to run

```powershell
# First time only: install dependencies
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Every time: start the server (loads .env automatically)
.\start.ps1
```

The server starts at `http://0.0.0.0:8000`. The Telegram bot starts automatically if `TELEGRAM_TOKEN` is set in `.env`.

## Key files

| File | Purpose |
|---|---|
| `reel_summarizer.py` | Core pipeline: download → transcribe → OCR → summarise |
| `server.py` | FastAPI server + async job queue + worker thread |
| `bot.py` | Telegram bot (same process as server, starts via lifespan) |
| `start.ps1` | Launcher: loads `.env` and runs server with venv Python |
| `.env.example` | All config options documented with defaults |
| `urls.txt` | Input for the CLI mode (one URL per line) |

## Environment / config

Copy `.env.example` to `.env` and fill in:
- `TELEGRAM_TOKEN` — from @BotFather
- `TELEGRAM_ALLOWED_USERS` — your Telegram user ID (get from @userinfobot)
- `OLLAMA_NUM_GPU=0` — required on this machine (Vulkan OOM on GPU)

## API

```
POST /summarize   { "url": "https://instagram.com/reel/..." }  → 202 { job_id }
GET  /jobs/{id}   → { status, current_step, summary, elapsed_s, ... }
GET  /jobs        → list all jobs (filter: ?status=done|failed|pending|processing)
GET  /health      → server + model readiness
```

## CLI mode (bypasses the server)

```powershell
.\venv\Scripts\python.exe reel_summarizer.py -i urls.txt -o reel_summaries.txt --device cpu --ollama-num-gpu 0
```

## Run tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known issues / workarounds

- **CUDA/cuDNN mismatch**: EasyOCR's bundled PyTorch doesn't match the installed cuDNN. Workaround: `--device cpu` in CLI, hardcoded `device="cpu"` in `server.py:_load_models`.
- **Ollama Vulkan OOM**: Ollama tries to load models onto the GPU via Vulkan and runs out of VRAM. Workaround: `OLLAMA_NUM_GPU=0` in `.env` forces CPU inference.
- **Always use the venv Python**: `python server.py` uses system Python (missing packages). Use `.\start.ps1` or `.\venv\Scripts\python.exe server.py`.

## Hardware

Currently running on **MSI Katana 15 B13 V** (i7-13620H, RTX 4060 8 GB, 16 GB DDR5). Everything runs on CPU due to the CUDA/Vulkan issues above. Once the driver/library mismatch is resolved, GPU inference will significantly speed up Whisper and OCR.

## Adding a new Telegram bot command

1. Add a handler function `async def _cmd_name(update, ctx)` in `bot.py`
2. Register it in `start()`: `_app.add_handler(CommandHandler("name", _cmd_name))`
3. Restart the server

## Phase roadmap

- [x] Phase 1: FastAPI server + async job queue
- [x] Phase 2: Telegram bot
- [ ] Phase 3: Tailscale setup (reach server from phone on any network)
- [ ] Phase 4: Fix GPU inference (cuDNN / Vulkan issues)
