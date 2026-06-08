# Local Instagram Reel Summarizer

Downloads public Instagram Reels, transcribes the speech, reads the on-screen
text, and writes a short bullet-point summary of each one — **fully on your
own machine**. No login, no API keys, no cloud services.

For each reel it produces an entry like this:

```
================================================================
URL: https://www.instagram.com/reel/XXXX/
----------------------------------------------------------------
- Main topic of the reel...
- Tools / techniques mentioned...
- Key claims and steps...
```

## How it works

1. **Download** — `yt-dlp` fetches each public reel (no account needed).
2. **Speech-to-text** — `faster-whisper` transcribes the audio locally.
3. **On-screen text** — frames are sampled (every ~2 s) and `easyocr` reads
   any captions, screenshots, or code shown on screen. Near-identical frames
   are skipped to save time.
4. **Summarize** — the merged transcript + on-screen text is sent to a local
   LLM running in **Ollama** (`localhost:11434`).
5. The summary is appended to `reel_summaries.txt` and the video is deleted.

## A note on "fully offline"

The processing is 100% local — no cloud, no credentials. There are only two
kinds of network traffic, and both are expected:

- **The reel download itself** (yt-dlp contacting Instagram) — this is the
  whole point of the tool.
- **One-time model downloads during setup** — Whisper, EasyOCR, and the
  Ollama model each download once on first use, then run offline forever.
  You can pre-download them, then set `HF_HUB_OFFLINE=1` if you want to be
  certain nothing reaches out afterward.

After setup, the tool never "phones home": no telemetry, no API keys.

## Requirements

- **Python 3.9+**
- **~3–4 GB free disk** for the Python packages (EasyOCR bundles PyTorch) plus
  model files.
- **Ollama** — a free local LLM runner.
- A GPU is optional. Everything runs on CPU; just pick smaller models.

## Setup

### 1. Install the Python packages

```bash
pip install -r requirements.txt
```

CPU-only laptops can save space by installing the CPU-only PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

Download Ollama from <https://ollama.com/download> (Windows / macOS / Linux),
install it, then pull a small model:

```bash
ollama pull mistral
```

Lighter alternatives for slow machines: `ollama pull phi3` or
`ollama pull llama3.2`. Make sure Ollama is running (the desktop app starts it
automatically, or run `ollama serve`).

### 3. (Optional) ffmpeg

Not required for typical reels — yt-dlp downloads them as a single MP4 and
faster-whisper reads the audio directly. Install ffmpeg only if a download
fails with a "merge" error.

## Usage

Put one reel URL per line in `urls.txt`, then run:

```bash
python reel_summarizer.py -i urls.txt -o reel_summaries.txt
```

That's it. Watch the progress in the terminal; results stream into
`reel_summaries.txt` as each reel finishes.

### Common options

| Option | Default | What it does |
|---|---|---|
| `-i, --input` | `urls.txt` | Input file, one URL per line |
| `-o, --output` | `reel_summaries.txt` | Output file |
| `--whisper-model` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` — smaller is faster |
| `--ollama-model` | `mistral` | Any model you've pulled in Ollama |
| `--device` | `auto` | `auto`/`cpu`/`cuda` for Whisper + OCR |
| `--frame-interval` | `2.0` | Seconds between sampled frames for OCR |
| `--max-frames` | `60` | Cap on frames OCR'd per reel |
| `--lang` | `en` | OCR language(s), e.g. `en,es` |
| `--keep-temp` | off | Keep downloaded videos instead of deleting |
| `--no-resume` | off | Regenerate everything (overwrite output) |
| `--preflight-only` | off | Check setup without downloading or processing reels |
| `--download-attempts` | `2` | Number of download attempts per reel before skipping |
| `--download-timeout` | `30` | Download socket timeout in seconds |
| `--retry-delay` | `2.0` | Seconds between download attempts |

Example — faster run on a slow laptop:

```bash
python reel_summarizer.py -i urls.txt --whisper-model tiny --ollama-model phi3 --frame-interval 3
```

Example setup check before a real run:

```bash
python reel_summarizer.py -i urls.txt --preflight-only
```

### Resume

By default the tool **skips URLs already present in the output file**, so you
can stop it (Ctrl+C) and re-run later — finished reels are saved. To start
fresh, use `--no-resume` or delete the output file.

## Example run

A live walk-through and a sample of the exact output format are in
**`example_output.txt`**. (A genuine end-to-end run can't be shown here
because it requires reaching Instagram, downloading models, and a running
Ollama instance — all of which happen on *your* machine.)

Typical console output:

```
3 URL(s) in input | 0 already done | 3 to process.
Whisper    : model='base', device=cpu (int8)
OCR        : easyocr, gpu=False, lang=en
Summarizer : Ollama model='mistral' at http://localhost:11434
Loading Whisper model (first run downloads it once)...
Loading EasyOCR model (first run downloads it once)...

[1/3] https://www.instagram.com/reel/EXAMPLE1/
  downloading...
  transcribing audio...
    transcript: 612 chars
  extracting on-screen text (OCR)...
    on-screen text: 248 chars
  summarizing with Ollama...
  done in 41s
...
Finished. Summaries written to: reel_summaries.txt
Run summary: 3/3 attempted, 2 succeeded, 1 skipped, 0 failed, 0 retry attempt(s).
```

## Troubleshooting

- **"Could not reach Ollama"** — start Ollama (open the app, or `ollama serve`)
  and check `--ollama-host`.
- **"model not found"** — run `ollama pull <model>` for the model you chose.
- **"missing Python package(s)"** — install the dependencies with
  `pip install -r requirements.txt`.
- **A reel is skipped / "content not available"** — the link is private,
  deleted, or Instagram is rate-limiting anonymous downloads. The tool reports
  every skipped URL at the end and continues with the rest. Since this works
  without a login, some reels simply won't be reachable — that's expected.
- **Temporary network failure** — raise `--download-attempts` or
  `--download-timeout` to give downloads more chances before a URL is skipped.
- **Downloads suddenly stop working** — Instagram changes often; update
  yt-dlp: `pip install -U yt-dlp`.
- **Too slow on CPU** — use `--whisper-model tiny`, a smaller Ollama model
  (`phi3`), and a larger `--frame-interval` (e.g. `3` or `4`).

## Files

| File | Purpose |
|---|---|
| `reel_summarizer.py` | The main script |
| `requirements.txt` | Python dependencies |
| `urls.txt` | Your list of reel URLs (edit this) |
| `example_output.txt` | Illustrative sample of the output format |
| `README.md` | This file |
