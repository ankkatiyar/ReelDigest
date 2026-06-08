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
            "- No speech or on-screen text could be extracted from this reel.",
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
