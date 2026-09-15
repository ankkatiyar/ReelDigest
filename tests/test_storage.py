import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import storage


def _vec(cosine_to_doc: float) -> np.ndarray:
    """A unit vector whose dot product with [1, 0] is exactly cosine_to_doc."""
    return np.array(
        [cosine_to_doc, math.sqrt(max(0.0, 1 - cosine_to_doc ** 2))], dtype=np.float32
    )


DOC = np.array([1.0, 0.0], dtype=np.float32)


def _job(job_id, summary, chat_id=1, transcript="", url=None):
    return types.SimpleNamespace(
        job_id=job_id,
        url=url or f"https://www.instagram.com/reel/{job_id}/",
        status="done",
        summary=summary,
        error=None,
        elapsed_s=1.0,
        retries_used=0,
        created_at="2026-01-01T00:00:00",
        started_at="2026-01-01T00:00:01",
        finished_at="2026-01-01T00:00:02",
        _transcript=transcript,
        _ocr_text="",
        _chat_id=chat_id,
    )


class FtsQueryTest(unittest.TestCase):
    def test_drops_stopwords_and_quotes_terms(self):
        self.assertEqual(
            storage._fts_query("is there something about the Postiz tool"),
            '"postiz" OR "tool"',
        )

    def test_small_talk_yields_no_query(self):
        for chatter in ("how are you", "hi", "what can you do for me"):
            self.assertEqual(storage._fts_query(chatter), "")

    def test_fts_operators_are_neutralised(self):
        # Quoting keeps FTS5 syntax in user text from changing the query.
        self.assertEqual(storage._fts_query('trading AND "drop table"'), '"trading" OR "drop" OR "table"')


class SearchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        with patch.object(storage, "embed", return_value=DOC):
            storage.init(str(root / "t.db"), str(root / "t.csv"))
            storage.persist(_job("aaa", "a video about algo trading", transcript="postiz"))
            storage.persist(_job("bbb", "a video about web scraping"))

    def test_returns_match_above_semantic_floor(self):
        with patch.object(storage, "embed", return_value=_vec(0.9)):
            results = storage.search(1, "algo trading")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["matched_by"], "keyword+semantic")

    def test_small_talk_returns_nothing(self):
        with patch.object(storage, "embed", return_value=_vec(0.3)):
            self.assertEqual(storage.search(1, "how are you"), [])

    def test_keyword_hit_below_keyword_floor_is_dropped(self):
        # FTS matches "postiz" in the transcript, but the topic is unrelated,
        # so the weak cosine must veto it.
        with patch.object(storage, "embed", return_value=_vec(0.4)):
            self.assertEqual(storage.search(1, "postiz"), [])

    def test_keyword_hit_between_floors_is_kept(self):
        between = (storage.KEYWORD_FLOOR + storage.SEMANTIC_FLOOR) / 2
        with patch.object(storage, "embed", return_value=_vec(between)):
            results = storage.search(1, "postiz")
        self.assertEqual([r["matched_by"] for r in results], ["keyword"])

    def test_other_chats_rows_are_not_searchable(self):
        with patch.object(storage, "embed", return_value=DOC):
            storage.persist(_job("ccc", "private to chat 2", chat_id=2))
        with patch.object(storage, "embed", return_value=_vec(0.9)):
            urls = [r["url"] for r in storage.search(1, "private")]
        self.assertNotIn("https://www.instagram.com/reel/ccc/", urls)

    def test_embedding_outage_falls_back_to_keyword_only(self):
        # Ollama down: with no query vector there is no semantic signal, so
        # keyword matches are served on their own rather than returning nothing.
        with patch.object(storage, "embed", return_value=None):
            results = storage.search(1, "algo trading")
        self.assertEqual([r["matched_by"] for r in results], ["keyword"])

    def test_embedding_outage_still_ignores_small_talk(self):
        with patch.object(storage, "embed", return_value=None):
            self.assertEqual(storage.search(1, "how are you"), [])

    def test_blank_query_returns_nothing(self):
        self.assertEqual(storage.search(1, "   "), [])


class BackfillTest(unittest.TestCase):
    def test_backfill_embeds_rows_saved_without_a_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(storage, "embed", return_value=None):
                storage.init(str(root / "t.db"), str(root / "t.csv"))
                storage.persist(_job("aaa", "a video about web scraping"))

            # Saved during an outage. A paraphrase shares no words with the
            # summary, so only a vector could match it — and there isn't one.
            paraphrase = "harvesting data from websites"
            with patch.object(storage, "embed", return_value=_vec(0.9)):
                self.assertEqual(storage.search(1, paraphrase), [])

            with patch.object(storage, "embed", return_value=DOC):
                self.assertEqual(storage.backfill_embeddings(), 1)

            with patch.object(storage, "embed", return_value=_vec(0.9)):
                results = storage.search(1, paraphrase)
            self.assertEqual([r["matched_by"] for r in results], ["semantic"])


if __name__ == "__main__":
    unittest.main()
