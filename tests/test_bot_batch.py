import asyncio
import sys
import threading
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import bot

A = "https://www.instagram.com/reel/AAA/"
B = "https://www.instagram.com/reel/BBB/"
C = "https://www.instagram.com/reel/CCC/"


def _fake_server(active=(), ready=True, error=None):
    submitted = []
    models_ready = threading.Event()
    if ready:
        models_ready.set()

    def submit(url, notify_fn=None, chat_id=None):
        submitted.append(url)
        return types.SimpleNamespace(job_id=f"job{len(submitted)}", queue_position=len(submitted) - 1)

    server = types.SimpleNamespace(
        _submit_job=submit,
        _find_active_job=lambda url, chat_id: url in active,
        _models_ready=models_ready,
        _models_error=error,
    )
    return server, submitted


def _update(text):
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = 1
    return update


class QueueBatchTest(unittest.TestCase):
    def _run(self, text, server, archived=None):
        archived = archived or {}
        update = _update(text)
        storage = types.SimpleNamespace(find_done=lambda chat_id, url: archived.get(url))
        with patch.dict(sys.modules, {"server": server, "storage": storage}), \
             patch.object(bot, "_is_allowed", return_value=True), \
             patch.dict(bot._user_jobs, clear=True):
            asyncio.run(bot._on_message(update, None))
        return [c.args[0] for c in update.message.reply_text.call_args_list]

    def test_sorts_links_into_new_archived_and_in_queue(self):
        server, submitted = _fake_server(active={C})
        saved = {"summary": "old summary", "finished_at": "2026-10-01T10:00:00"}
        replies = self._run(f"{A}\n{B}\n{C}", server, archived={B: saved})

        self.assertEqual(submitted, [A])
        self.assertIn("Got 3 links", replies[0])
        self.assertIn("Queued 1", replies[0])
        self.assertIn("1 already in your archive", replies[0])
        self.assertIn("1 already in the queue", replies[0])
        self.assertEqual(len(replies), 2)
        self.assertIn("old summary", replies[1])
        self.assertIn("2026-10-01", replies[1])

    def test_nothing_is_queued_while_models_load(self):
        server, submitted = _fake_server(ready=False)
        replies = self._run(f"{A} {B}", server)
        self.assertEqual(submitted, [])
        self.assertIn("2 not queued", replies[0])

    def test_single_link_keeps_the_one_link_reply(self):
        server, submitted = _fake_server()
        replies = self._run(f"look {A}", server)
        self.assertEqual(submitted, [A])
        self.assertIn("Got it!", replies[0])


if __name__ == "__main__":
    unittest.main()
