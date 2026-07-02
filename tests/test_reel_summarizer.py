import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import reel_summarizer as rs


class UrlHelpersTest(unittest.TestCase):
    def test_load_urls_skips_comments_blanks_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "urls.txt"
            path.write_text(
                "\n"
                "# comment\n"
                "https://www.instagram.com/reel/ONE/\n"
                "https://www.instagram.com/reel/TWO/\n"
                "https://www.instagram.com/reel/ONE/\n",
                encoding="utf-8",
            )

            self.assertEqual(
                rs.load_urls(str(path)),
                [
                    "https://www.instagram.com/reel/ONE/",
                    "https://www.instagram.com/reel/TWO/",
                ],
            )

    def test_load_urls_missing_file_exits_cleanly(self):
        with self.assertRaises(SystemExit) as caught:
            rs.load_urls("missing-urls.txt")

        self.assertIn("Input file not found", str(caught.exception))

    def test_load_done_urls_handles_missing_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.txt"
            self.assertEqual(rs.load_done_urls(str(missing)), set())

            output = Path(tmp) / "summaries.txt"
            output.write_text(
                "# generated\n\n"
                "================================================================\n"
                "URL: https://www.instagram.com/reel/ONE/\n"
                "----------------------------------------------------------------\n"
                "- summary\n",
                encoding="utf-8",
            )

            self.assertEqual(
                rs.load_done_urls(str(output)),
                {"https://www.instagram.com/reel/ONE/"},
            )


class SourceUrlTest(unittest.TestCase):
    def test_detect_platform(self):
        self.assertEqual(
            rs.detect_platform("https://www.instagram.com/reel/ABC/"), "instagram"
        )
        self.assertEqual(
            rs.detect_platform("https://youtu.be/ABC"), "youtube"
        )
        self.assertEqual(
            rs.detect_platform("https://www.youtube.com/watch?v=ABC"), "youtube"
        )
        self.assertIsNone(rs.detect_platform("https://example.com/x"))

    def test_is_supported_url_accepts_posts_and_videos(self):
        for url in (
            "https://www.instagram.com/reel/ABC/",
            "https://www.instagram.com/p/ABC/",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/ABC123",
        ):
            self.assertTrue(rs.is_supported_url(url), url)

    def test_is_supported_url_rejects_profiles_and_junk(self):
        for url in (
            "https://www.instagram.com/some_user/",       # profile, not a post
            "https://www.youtube.com/@somechannel",       # channel, not a video
            "https://example.com/watch?v=ABC",
            "not a url",
        ):
            self.assertFalse(rs.is_supported_url(url), url)

    def test_extract_url_pulls_link_from_text(self):
        self.assertEqual(
            rs.extract_url("check this out https://youtu.be/ABC123 cool right?"),
            "https://youtu.be/ABC123",
        )
        self.assertEqual(
            rs.extract_url(
                "https://www.instagram.com/reel/XYZ/?igsh=abc123 nice"
            ),
            "https://www.instagram.com/reel/XYZ/?igsh=abc123",
        )
        self.assertIsNone(rs.extract_url("no links here"))


class VttCaptionTest(unittest.TestCase):
    def test_parse_vtt_cleans_and_dedups_rolling_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.en.vtt"
            path.write_text(
                "WEBVTT\n"
                "Kind: captions\n"
                "Language: en\n"
                "\n"
                "00:00:00.000 --> 00:00:02.000\n"
                "Hello and welcome\n"
                "\n"
                "00:00:02.000 --> 00:00:04.000\n"
                "Hello and welcome to the show\n"          # supersedes the line above
                "\n"
                "00:00:04.000 --> 00:00:06.000\n"
                "<c>today we discuss</c> stocks\n",         # inline tags stripped
                encoding="utf-8",
            )
            self.assertEqual(
                rs._parse_vtt(str(path)),
                "Hello and welcome to the show today we discuss stocks",
            )

    def test_parse_vtt_empty_when_only_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.en.vtt"
            path.write_text("WEBVTT\n\n", encoding="utf-8")
            self.assertEqual(rs._parse_vtt(str(path)), "")


class OutputAndRetryTest(unittest.TestCase):
    def test_write_entry_writes_one_complete_block(self):
        out = io.StringIO()

        rs.write_entry(out, "https://example.test/reel", "- A summary")

        self.assertEqual(
            out.getvalue(),
            "================================================================\n"
            "URL: https://example.test/reel\n"
            "----------------------------------------------------------------\n"
            "- A summary\n\n",
        )

    def test_run_with_retries_returns_retry_count(self):
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary")
            return "ok"

        value, retries = rs.run_with_retries(
            flaky, attempts=2, delay=0, label="test"
        )

        self.assertEqual(value, "ok")
        self.assertEqual(retries, 1)

    def test_run_with_retries_marks_final_exception(self):
        with self.assertRaises(RuntimeError) as caught:
            rs.run_with_retries(
                lambda: (_ for _ in ()).throw(RuntimeError("still bad")),
                attempts=3,
                delay=0,
                label="test",
            )

        self.assertEqual(getattr(caught.exception, "_retries_used"), 2)


class OllamaAndSummaryTest(unittest.TestCase):
    def test_summarize_empty_extraction_returns_local_message(self):
        self.assertEqual(
            rs.summarize("", "", "mistral", "http://localhost:11434", 1000),
            "- No speech or on-screen text could be extracted from this post.",
        )

    def test_check_ollama_unavailable_exits_once(self):
        fake_requests = types.SimpleNamespace(
            get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("connection refused")
            )
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaises(SystemExit) as caught:
                rs.check_ollama("http://localhost:11434", "mistral")

        self.assertIn("could not reach Ollama", str(caught.exception))

    def test_check_ollama_missing_model_can_be_fatal_or_warning(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "phi3:latest"}]}

        fake_requests = types.SimpleNamespace(
            get=lambda *_args, **_kwargs: Response()
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaises(SystemExit) as caught:
                rs.check_ollama("http://localhost:11434", "mistral")
            self.assertIn("model 'mistral' was not found", str(caught.exception))

            rs.check_ollama(
                "http://localhost:11434",
                "mistral",
                require_model=False,
            )


class PipelineTest(unittest.TestCase):
    def _args(self, temp_dir):
        return types.SimpleNamespace(
            temp_dir_path=temp_dir,
            ytdlp_retries=1,
            download_timeout=1,
            download_attempts=1,
            retry_delay=0,
            frame_interval=2.0,
            max_frames=5,
            diff_threshold=8.0,
            ollama_model="mistral",
            ollama_host="http://localhost:11434",
            ollama_num_gpu=-1,
            max_chars=1000,
            keep_temp=False,
        )

    def test_process_url_records_download_failure_as_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with patch.object(
                rs, "download_reel", side_effect=RuntimeError("private reel")
            ):
                result = rs.process_url(
                    "https://www.instagram.com/reel/ONE/",
                    out,
                    whisper_model=None,
                    ocr_reader=None,
                    args=self._args(tmp),
                    index=1,
                    total=1,
                )

            self.assertEqual(result.status, "skipped")
            self.assertIn("[SKIPPED: download failed - private reel]", out.getvalue())

    def test_process_url_records_processing_failure_and_removes_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "reel.mp4"
            video.write_bytes(b"fake video")
            out = io.StringIO()

            with patch.object(rs, "download_reel", return_value=[str(video)]):
                with patch.object(rs, "transcribe", return_value="hello"):
                    with patch.object(rs, "extract_media_text", return_value="text"):
                        with patch.object(
                            rs, "summarize", side_effect=RuntimeError("ollama down")
                        ):
                            result = rs.process_url(
                                "https://www.instagram.com/reel/ONE/",
                                out,
                                whisper_model=None,
                                ocr_reader=None,
                                args=self._args(tmp),
                                index=1,
                                total=1,
                            )

            self.assertEqual(result.status, "failed")
            self.assertFalse(os.path.exists(video))
            self.assertIn("[FAILED: processing failed - ollama down]", out.getvalue())


if __name__ == "__main__":
    unittest.main()
