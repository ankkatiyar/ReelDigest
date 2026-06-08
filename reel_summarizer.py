#!/usr/bin/env python3
"""
reel_summarizer.py - Fully local Instagram Reel summarizer.

For each public Instagram Reel URL in an input file, this script:
  1. Downloads the reel video with yt-dlp (no login, public reels only).
  2. Transcribes the speech with faster-whisper (local model).
  3. Samples frames and extracts on-screen text with EasyOCR (local model).
  4. Merges the transcript + on-screen text.
  5. Summarizes it into bullet points with a local LLM served by Ollama.
  6. Appends the summary to the output file and deletes the temp video.

No cloud APIs, no API keys, no login. The ONLY network traffic is the reel
download itself (from Instagram) and one-time model downloads during initial
setup. After setup, everything runs offline on localhost.

Usage:
    python reel_summarizer.py -i urls.txt -o reel_summaries.txt

Run `python reel_summarizer.py --help` for all options.
"""

import argparse
import difflib
import glob
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime

# NOTE: heavy dependencies (faster_whisper, easyocr, cv2, yt_dlp, requests)
# are imported lazily inside the functions that need them. This keeps startup
# fast and lets `--help` work before everything is installed.

PROMPT_TEMPLATE = """You summarize Instagram posts (Reels and carousel posts) on any topic.

Below is the automatically extracted spoken transcript and on-screen text. For carousel posts with multiple slides the on-screen text from each slide is separated by "---". The extraction can be noisy, duplicated, or contain recognition errors - use your judgment and ignore obvious noise.

Write the summary as 3-6 short bullet points (start each line with "- ") covering: the main topic, any tools / products / people / techniques named, the key claims or advice, and any concrete steps shown. Be specific and factual. Do NOT invent details that are not present in the text. If the content is too sparse to summarize meaningfully, say that in a single bullet.

=== SPOKEN TRANSCRIPT ===
{transcript}

=== ON-SCREEN TEXT ===
{ocr}

=== SUMMARY ==="""


REQUIRED_MODULES = {
    "yt_dlp": "yt-dlp",
    "faster_whisper": "faster-whisper",
    "easyocr": "easyocr",
    "cv2": "opencv-python-headless",
    "requests": "requests",
}


@dataclass
class ProcessResult:
    """Outcome for one URL. Kept small so a future UI can reuse it."""
    url: str
    status: str
    reason: str = ""
    elapsed: float = 0.0
    retries: int = 0


@dataclass
class RunStats:
    total: int
    already_done: int
    pending: int
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    retried: int = 0
    interrupted: bool = False
    skipped_items: list = field(default_factory=list)
    failed_items: list = field(default_factory=list)

    def record(self, result):
        self.retried += result.retries
        if result.status == "success":
            self.succeeded += 1
        elif result.status == "skipped":
            self.skipped += 1
            self.skipped_items.append((result.url, result.reason))
        else:
            self.failed += 1
            self.failed_items.append((result.url, result.reason))


def log(msg=""):
    print(msg, flush=True)


def short_reason(exc, max_len=200):
    """Return a one-line error reason that is useful in logs and output."""
    return str(exc).splitlines()[0][:max_len] or exc.__class__.__name__


def flush_output(out):
    """Flush and fsync when possible so completed entries survive crashes."""
    out.flush()
    try:
        os.fsync(out.fileno())
    except (AttributeError, io.UnsupportedOperation, OSError, ValueError):
        pass


def find_missing_dependencies():
    """Return installed-package names missing from the current environment."""
    missing = []
    for module, package in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    return missing


def check_dependencies():
    missing = find_missing_dependencies()
    if missing:
        raise SystemExit(
            "\nERROR: missing Python package(s): "
            + ", ".join(missing)
            + "\nInstall them with:  pip install -r requirements.txt"
        )


def run_with_retries(action, attempts, delay, label):
    """Run an action with simple retries. Returns (value, retries_used)."""
    attempts = max(1, attempts)
    retries_used = 0
    for attempt in range(1, attempts + 1):
        try:
            return action(), retries_used
        except Exception as exc:
            if attempt >= attempts:
                setattr(exc, "_retries_used", retries_used)
                raise
            retries_used += 1
            log(f"    ! {label} failed ({short_reason(exc)}); "
                f"retrying {attempt}/{attempts - 1}...")
            if delay > 0:
                time.sleep(delay)


# --------------------------------------------------------------------------
# Input / output helpers
# --------------------------------------------------------------------------

def load_urls(path):
    """Read URLs from a text file, one per line. Skips blanks, '#' comments
    and duplicates."""
    if not os.path.exists(path):
        raise SystemExit(f"Input file not found: {path}")
    urls, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line not in seen:
                seen.add(line)
                urls.append(line)
    return urls


def load_done_urls(path):
    """Return the set of URLs that already have an entry in the output file
    (used for resume support)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("URL:"):
                done.add(line[4:].strip())
    return done


def write_entry(out, url, body):
    """Append one URL + summary block to the output file and flush, so
    progress survives a crash or Ctrl+C."""
    block = (
        "=" * 64 + "\n"
        + f"URL: {url}\n"
        + "-" * 64 + "\n"
        + body.strip() + "\n\n"
    )
    out.write(block)
    flush_output(out)


# --------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------

def resolve_device(choice):
    """Return (device, compute_type) for faster-whisper."""
    if choice == "cpu":
        return "cpu", "int8"
    if choice == "cuda":
        return "cuda", "float16"
    # auto-detect
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# --------------------------------------------------------------------------
# Step 1: download
# --------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def is_image(path):
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _fetch_instagram_images(url, temp_dir, cookies_file, timeout=30):
    """Download image-only carousel or photo posts via Instagram's internal API.

    Called when yt-dlp raises 'No video formats found!' (image-only posts that
    yt-dlp's extractor can't handle).  Requires a valid cookies_file.
    """
    import re as _re
    import requests
    from http.cookiejar import MozillaCookieJar

    m = _re.search(r'instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)', url)
    if not m:
        raise ValueError(f"Cannot extract shortcode from URL: {url}")
    shortcode = m.group(1)

    # Shortcode → numeric media ID (Instagram's base-64 encoding)
    _ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    media_id = 0
    for c in shortcode:
        media_id = media_id * 64 + _ALPHA.index(c)

    jar = MozillaCookieJar()
    jar.load(cookies_file, ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies = jar
    session.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/131.0.0.0 Safari/537.36'
        ),
        'X-IG-App-ID': '936619743392459',
        'Accept': '*/*',
        'Referer': 'https://www.instagram.com/',
    })

    resp = session.get(
        f'https://www.instagram.com/api/v1/media/{media_id}/info/',
        timeout=timeout,
    )
    resp.raise_for_status()
    items = resp.json().get('items', [])
    if not items:
        raise FileNotFoundError("Instagram API returned no media for this post")

    item = items[0]
    # carousel_media for multi-slide posts; wrap single images in a list
    slides = item.get('carousel_media') or [item]

    paths = []
    for i, slide in enumerate(slides):
        candidates = slide.get('image_versions2', {}).get('candidates', [])
        if not candidates:
            continue
        img_url = candidates[0]['url']   # first candidate = highest resolution
        dest = os.path.join(temp_dir, f"{shortcode}_{i:02d}.jpg")
        img_resp = session.get(img_url, timeout=timeout)
        img_resp.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(img_resp.content)
        log(f"    slide {i + 1}/{len(slides)}: {os.path.basename(dest)}")
        paths.append(dest)

    if not paths:
        raise FileNotFoundError("No images found in Instagram API response")
    return paths


def download_reel(url, temp_dir, retries=3, socket_timeout=30, cookies_file=None):
    """Download a reel or every slide of a carousel post.
    Returns a list of local file paths (one for reels, multiple for carousels).

    Uses yt-dlp for video reels/carousels.  Falls back to a direct Instagram
    API call for image-only carousel posts (yt-dlp raises 'No video formats
    found!' for those due to a known extractor limitation).

    cookies_file: Netscape-format cookies.txt from a browser logged into
    Instagram.  Required — Meta disabled anonymous API access.
    """
    from yt_dlp import YoutubeDL

    before = set(os.listdir(temp_dir))
    _dl_errors: list[str] = []

    class _Logger:
        def debug(self, msg): pass
        def warning(self, msg): log(f"    yt-dlp: {msg}")
        def error(self, msg):
            log(f"    yt-dlp ERROR: {msg}")
            _dl_errors.append(msg)

    def _base_opts():
        opts = {
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": False,
            "retries": retries,
            "socket_timeout": socket_timeout,
            "logger": _Logger(),
        }
        if cookies_file and os.path.isfile(cookies_file):
            opts["cookiefile"] = cookies_file
        elif cookies_file:
            log(f"    WARNING: cookies file not found: {cookies_file}")
        return opts

    log(f"    downloading: {url}")
    if cookies_file and os.path.isfile(cookies_file):
        log(f"    cookies: {cookies_file}")

    # Stage 1: yt-dlp — works for video reels and video carousels
    try:
        opts = _base_opts()
        opts["format"] = "best"
        with YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        if "No video formats found" not in str(exc):
            raise
        # Stage 2: yt-dlp can't handle image-only posts; use Instagram's API directly
        log("    no video found; downloading images via Instagram API...")
        if not (cookies_file and os.path.isfile(cookies_file)):
            raise FileNotFoundError(
                "Image carousel detected but no cookies file available. "
                "Set INSTAGRAM_COOKIES_FILE in your .env (see .env.example)."
            )
        _fetch_instagram_images(url, temp_dir, cookies_file, timeout=socket_timeout)

    after = set(os.listdir(temp_dir))
    new_files = sorted(
        f for f in (after - before)
        if not f.endswith((".part", ".ytdl", ".tmp"))
    )

    if not new_files:
        if _dl_errors:
            # Strip the verbose yt-dlp prefix for a cleaner error message
            msg = _dl_errors[0]
            for prefix in ("[Instagram] ", "ERROR: "):
                msg = msg.replace(prefix, "")
            raise FileNotFoundError(msg.strip())
        raise FileNotFoundError(
            "Instagram returned no media. "
            "Set INSTAGRAM_COOKIES_FILE in your .env (see .env.example)."
        )

    paths = [os.path.join(temp_dir, f) for f in new_files]
    log(f"    found {len(paths)} file(s): {', '.join(new_files)}")
    return paths


# --------------------------------------------------------------------------
# Step 2: transcription
# --------------------------------------------------------------------------

def transcribe(model, video_path):
    """Transcribe speech from the video. faster-whisper reads the audio
    stream directly from the mp4. Returns '' if there is no speech."""
    try:
        segments, _info = model.transcribe(
            video_path, beam_size=5, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log(f"    ! transcription failed ({e}); continuing without audio text")
        return ""


# --------------------------------------------------------------------------
# Step 3: on-screen text (OCR)
# --------------------------------------------------------------------------

def _add_unique(collected, text, ratio=0.9):
    """Add `text` to `collected`, merging near-duplicates. Handles the common
    case of progressively-revealed captions (one line being a superset of
    another)."""
    norm = " ".join(text.split())
    if not norm:
        return
    low = norm.lower()
    for i, existing in enumerate(collected):
        el = existing.lower()
        if low in el:                       # already covered
            return
        if el in low:                       # new line supersedes old
            collected[i] = norm
            return
        if difflib.SequenceMatcher(None, low, el).ratio() >= ratio:
            if len(norm) > len(existing):
                collected[i] = norm
            return
    collected.append(norm)


def extract_and_ocr(video_path, reader, interval, max_frames, diff_threshold):
    """Sample a frame every `interval` seconds, skip frames that are nearly
    identical to the previous OCR'd frame, and run OCR on the rest. Returns
    the de-duplicated on-screen text."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval)))

    collected, last_small = [], None
    frame_no, ocr_count = 0, 0
    try:
        while ocr_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % step == 0:
                small = cv2.cvtColor(
                    cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY
                )
                changed = (
                    last_small is None
                    or cv2.absdiff(small, last_small).mean() > diff_threshold
                )
                if changed:
                    last_small = small
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for t in reader.readtext(rgb, detail=0, paragraph=True):
                        _add_unique(collected, t)
                    ocr_count += 1
            frame_no += 1
    finally:
        cap.release()

    return "\n".join(collected)


def ocr_image(path, reader):
    """Run OCR directly on a single image file (carousel image slides)."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return ""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    collected = []
    for t in reader.readtext(rgb, detail=0, paragraph=True):
        _add_unique(collected, t)
    return "\n".join(collected)


def extract_media_text(path, reader, interval, max_frames, diff_threshold):
    """Dispatch to image OCR or video frame OCR based on file type."""
    if is_image(path):
        return ocr_image(path, reader)
    return extract_and_ocr(path, reader, interval, max_frames, diff_threshold)


# --------------------------------------------------------------------------
# Step 4: summarization via Ollama
# --------------------------------------------------------------------------

def check_ollama(host, model, require_model=True):
    """Pre-flight check so we fail fast (once) instead of failing on every
    URL if Ollama is not running."""
    import requests

    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"\nERROR: could not reach Ollama at {host}\n"
            f"Start it first ('ollama serve', or just open the Ollama app), "
            f"then re-run.\nDetails: {e}"
        )

    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(n == model or n.startswith(model + ":") for n in names):
        message = (
            f"\nERROR: Ollama model '{model}' was not found.\n"
            f"Pull it with:  ollama pull {model}\n"
            f"Installed models: {', '.join(names) or '(none)'}"
        )
        if require_model:
            raise SystemExit(message)
        log("WARNING: " + message.strip().replace("\n", "\n         "))


def summarize(transcript, ocr_text, model, host, max_chars, num_gpu=-1):
    """Send the merged text to Ollama and return a bullet-point summary."""
    t = transcript.strip()[:max_chars]
    o = ocr_text.strip()[:max_chars]
    if not t and not o:
        return "- No speech or on-screen text could be extracted from this reel."

    import requests

    options = {"temperature": 0.2}
    if num_gpu >= 0:
        options["num_gpu"] = num_gpu

    prompt = PROMPT_TEMPLATE.format(transcript=t or "(none)", ocr=o or "(none)")
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        timeout=600,
    )
    resp.raise_for_status()
    summary = resp.json().get("response", "").strip()
    return summary or "- (The model returned an empty summary.)"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def process_url(url, out, whisper_model, ocr_reader, args, index, total):
    """Process one URL end-to-end and write exactly one output entry.
    Handles both single reels and multi-slide carousel posts."""
    t0 = time.time()
    log(f"\n[{index}/{total}] {url}")
    media_paths = []
    retries_used = 0

    try:
        log("  downloading...")
        media_paths, retries_used = run_with_retries(
            lambda: download_reel(
                url,
                args.temp_dir_path,
                retries=args.ytdlp_retries,
                socket_timeout=args.download_timeout,
            ),
            attempts=args.download_attempts,
            delay=args.retry_delay,
            label="download",
        )
        slide_label = f"{len(media_paths)} slide(s)" if len(media_paths) > 1 else "1 file"
        log(f"    downloaded {slide_label}")
    except Exception as exc:
        retries_used = getattr(exc, "_retries_used", retries_used)
        reason = short_reason(exc)
        log(f"  ! skipped (download failed): {reason}")
        write_entry(out, url, f"[SKIPPED: download failed - {reason}]")
        return ProcessResult(
            url=url,
            status="skipped",
            reason=reason,
            elapsed=time.time() - t0,
            retries=retries_used,
        )

    try:
        all_transcripts, all_ocr = [], []

        for path in media_paths:
            if not is_image(path):
                t = transcribe(whisper_model, path)
                if t:
                    all_transcripts.append(t)

            o = extract_media_text(
                path, ocr_reader, args.frame_interval,
                args.max_frames, args.diff_threshold,
            )
            if o:
                all_ocr.append(o)

        transcript = " ".join(all_transcripts)
        ocr_text   = "\n---\n".join(all_ocr)
        log(f"    transcript: {len(transcript)} chars | on-screen text: {len(ocr_text)} chars")

        log("  summarizing with Ollama...")
        summary = summarize(
            transcript, ocr_text, args.ollama_model,
            args.ollama_host, args.max_chars, args.ollama_num_gpu,
        )
        write_entry(out, url, summary)
        elapsed = time.time() - t0
        log(f"  done in {elapsed:.0f}s")
        return ProcessResult(
            url=url,
            status="success",
            elapsed=elapsed,
            retries=retries_used,
        )
    except Exception as exc:
        reason = short_reason(exc)
        log(f"  ! failed (processing failed): {reason}")
        write_entry(out, url, f"[FAILED: processing failed - {reason}]")
        return ProcessResult(
            url=url,
            status="failed",
            reason=reason,
            elapsed=time.time() - t0,
            retries=retries_used,
        )
    finally:
        if not args.keep_temp:
            for path in media_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def log_final_report(stats, output_path):
    processed = stats.succeeded + stats.skipped + stats.failed
    if stats.interrupted:
        log("\nInterrupted. Progress so far has been saved.")

    log(f"\nFinished. Summaries written to: {output_path}")
    log(
        "Run summary: "
        f"{processed}/{stats.pending} attempted, "
        f"{stats.succeeded} succeeded, "
        f"{stats.skipped} skipped, "
        f"{stats.failed} failed, "
        f"{stats.retried} retry attempt(s)."
    )

    if stats.skipped_items:
        log(f"\n{len(stats.skipped_items)} URL(s) were skipped:")
        for u, r in stats.skipped_items:
            log(f"  - {u}  ->  {r}")
        log("To retry skipped URLs, delete their entries from the output "
            "file (or use --no-resume) and run again.")

    if stats.failed_items:
        log(f"\n{len(stats.failed_items)} URL(s) failed during processing:")
        for u, r in stats.failed_items:
            log(f"  - {u}  ->  {r}")
        log("These were written as [FAILED] entries so the run can resume "
            "without repeating completed work.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize public Instagram Reels fully locally "
                    "(no credentials, no cloud).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default="urls.txt",
                   help="Text file with one reel URL per line.")
    p.add_argument("-o", "--output", default="reel_summaries.txt",
                   help="Where to write the summaries.")
    p.add_argument("--whisper-model", default="base",
                   help="faster-whisper model: tiny / base / small / medium / "
                        "large-v3 (smaller = faster).")
    p.add_argument("--ollama-model", default="mistral",
                   help="Local Ollama model used for summarization.")
    p.add_argument("--ollama-host", default="http://localhost:11434",
                   help="Base URL of the local Ollama server.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Compute device for Whisper / OCR.")
    p.add_argument("--lang", default="en",
                   help="OCR language code(s), comma-separated (e.g. en,es).")
    p.add_argument("--frame-interval", type=float, default=2.0,
                   help="Seconds between sampled frames for OCR.")
    p.add_argument("--max-frames", type=int, default=60,
                   help="Maximum number of frames to OCR per reel.")
    p.add_argument("--diff-threshold", type=float, default=8.0,
                   help="Skip a frame for OCR if it differs from the previous "
                        "one by less than this (0-255 mean pixel diff).")
    p.add_argument("--max-chars", type=int, default=6000,
                   help="Truncate transcript / OCR text fed to the LLM.")
    p.add_argument("--download-attempts", type=int, default=2,
                   help="How many times to try downloading each reel before "
                        "marking it skipped.")
    p.add_argument("--retry-delay", type=float, default=2.0,
                   help="Seconds to wait between download attempts.")
    p.add_argument("--download-timeout", type=int, default=30,
                   help="Socket timeout, in seconds, for yt-dlp downloads.")
    p.add_argument("--ytdlp-retries", type=int, default=3,
                   help="Internal yt-dlp retry count for each download "
                        "attempt.")
    p.add_argument("--temp-dir", default=None,
                   help="Folder for temporary downloads (default: a system "
                        "temp folder, auto-deleted).")
    p.add_argument("--keep-temp", action="store_true",
                   help="Keep downloaded videos instead of deleting them.")
    p.add_argument("--no-resume", action="store_true",
                   help="Regenerate from scratch (overwrite output) instead "
                        "of skipping already-summarized URLs.")
    p.add_argument("--preflight-only", action="store_true",
                   help="Check input, dependencies, device, and Ollama setup "
                        "without downloading or processing reels.")
    p.add_argument("--allow-missing-ollama-model", action="store_true",
                   help="Warn instead of stopping if the selected Ollama model "
                        "is not listed locally.")
    p.add_argument("--ollama-num-gpu", type=int, default=-1,
                   help="Number of GPU layers for Ollama (-1 = Ollama decides, "
                        "0 = CPU only). Use 0 if you get out-of-VRAM errors.")
    return p.parse_args()


def main():
    args = parse_args()

    urls = load_urls(args.input)
    if not urls:
        raise SystemExit(f"No URLs found in {args.input}.")

    resume = not args.no_resume
    done, out_mode = set(), "w"
    if resume and os.path.exists(args.output):
        done = load_done_urls(args.output)
        out_mode = "a"

    pending = [u for u in urls if u not in done]
    stats = RunStats(
        total=len(urls),
        already_done=len(done),
        pending=len(pending),
    )
    log(f"{len(urls)} URL(s) in input | {len(done)} already done | "
        f"{len(pending)} to process.")
    if not pending:
        log("Nothing to do. Use --no-resume to regenerate everything.")
        return

    check_dependencies()

    device, compute_type = resolve_device(args.device)
    use_gpu = device == "cuda"
    log(f"Whisper    : model='{args.whisper_model}', device={device} "
        f"({compute_type})")
    log(f"OCR        : easyocr, gpu={use_gpu}, lang={args.lang}")
    log(f"Summarizer : Ollama model='{args.ollama_model}' at {args.ollama_host}")

    check_ollama(
        args.ollama_host,
        args.ollama_model,
        require_model=not args.allow_missing_ollama_model,
    )

    if args.preflight_only:
        log("\nPreflight OK. Input, dependencies, device selection, and "
            "Ollama connection are ready.")
        return

    log("Loading Whisper model (first run downloads it once)...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        args.whisper_model, device=device, compute_type=compute_type
    )

    log("Loading EasyOCR model (first run downloads it once)...")
    import easyocr
    ocr_reader = easyocr.Reader(
        [s.strip() for s in args.lang.split(",") if s.strip()],
        gpu=use_gpu, verbose=False,
    )

    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="reels_")
    os.makedirs(temp_dir, exist_ok=True)
    args.temp_dir_path = temp_dir

    out = open(args.output, out_mode, encoding="utf-8")
    if out_mode == "w":
        out.write(f"# Instagram Reel Summaries - generated "
                  f"{datetime.now():%Y-%m-%d %H:%M}\n\n")
        flush_output(out)

    try:
        for i, url in enumerate(pending, 1):
            result = process_url(
                url, out, whisper_model, ocr_reader, args, i, len(pending)
            )
            stats.record(result)
    except KeyboardInterrupt:
        stats.interrupted = True
    finally:
        out.close()
        if not args.keep_temp and not args.temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    log_final_report(stats, args.output)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log(f"\nFatal error: {e}")
        sys.exit(1)
