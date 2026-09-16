import asyncio
import unittest
from unittest.mock import patch

from telegram.error import InvalidToken

import bot


class BotStateTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = (
            bot._app, bot._last_poll_error, bot._token, bot._last_start_error,
        )
        bot._last_poll_error = None
        bot._last_start_error = None
        bot._token = ""

        def restore():
            (
                bot._app, bot._last_poll_error, bot._token, bot._last_start_error,
            ) = self._saved

        self.addCleanup(restore)


class ConnectionStatusTest(BotStateTestCase):

    def test_reports_not_running_before_start(self):
        bot._app = None
        status = bot.connection_status()
        self.assertFalse(status["running"])
        self.assertFalse(status["connected"])

    def test_connected_when_no_polling_error_seen(self):
        bot._app = object()
        self.assertTrue(bot.connection_status()["connected"])

    def test_recent_error_reads_as_disconnected(self):
        bot._app = object()
        with patch.object(bot.time, "monotonic", return_value=1000.0):
            bot._on_poll_error(Exception("getaddrinfo failed"))
            status = bot.connection_status()
        self.assertFalse(status["connected"])
        self.assertIn("getaddrinfo failed", status["last_error"])

    def test_old_error_reads_as_reconnected(self):
        bot._app = object()
        with patch.object(bot.time, "monotonic", return_value=1000.0):
            bot._on_poll_error(Exception("getaddrinfo failed"))
        # Well past the backoff ceiling with no further errors: we are through.
        with patch.object(
            bot.time, "monotonic", return_value=1000.0 + bot._DISCONNECTED_AFTER + 1
        ):
            self.assertTrue(bot.connection_status()["connected"])

    def test_token_is_redacted_from_the_error(self):
        bot._app = object()
        bot._token = "123456:SECRET"
        bot._on_poll_error(Exception("failed calling /bot123456:SECRET/getUpdates"))
        self.assertNotIn("SECRET", bot.connection_status()["last_error"])


class StartWithRetryTest(BotStateTestCase):
    def test_retries_until_the_network_comes_up(self):
        attempts = []

        async def flaky(token):
            attempts.append(token)
            if len(attempts) < 3:
                raise OSError("getaddrinfo failed")
            bot._app = object()

        with patch.object(bot, "start", flaky), \
                patch.object(bot.asyncio, "sleep", side_effect=self._no_wait):
            asyncio.run(bot.start_with_retry("123:TOKEN"))

        self.assertEqual(len(attempts), 3)
        self.assertIsNone(bot._last_start_error)
        self.assertTrue(bot.connection_status()["connected"])

    def test_gives_up_immediately_on_an_invalid_token(self):
        attempts = []

        async def rejected(token):
            attempts.append(token)
            raise InvalidToken("bad token")

        with patch.object(bot, "start", rejected), \
                patch.object(bot.asyncio, "sleep", side_effect=self._no_wait):
            asyncio.run(bot.start_with_retry("123:TOKEN"))

        self.assertEqual(len(attempts), 1)
        self.assertIn("invalid", bot._last_start_error)

    def test_status_explains_why_it_is_not_running_yet(self):
        async def failing(token):
            raise OSError("network is unreachable")

        with patch.object(bot, "start", failing), \
                patch.object(bot.asyncio, "sleep", side_effect=self._stop_after_one):
            with self.assertRaises(_StopRetrying):
                asyncio.run(bot.start_with_retry("123:TOKEN"))

        status = bot.connection_status()
        self.assertFalse(status["running"])
        self.assertIn("unreachable", status["last_error"])

    def test_token_is_redacted_from_a_start_failure(self):
        async def failing(token):
            raise OSError("POST /bot123:SECRET/getMe failed")

        with patch.object(bot, "start", failing), \
                patch.object(bot.asyncio, "sleep", side_effect=self._stop_after_one):
            with self.assertRaises(_StopRetrying):
                asyncio.run(bot.start_with_retry("123:SECRET"))

        self.assertNotIn("SECRET", bot._last_start_error)

    @staticmethod
    async def _no_wait(_delay):
        return None

    @staticmethod
    async def _stop_after_one(_delay):
        raise _StopRetrying


class _StopRetrying(Exception):
    """Breaks the retry loop in tests without waiting on real backoff."""


if __name__ == "__main__":
    unittest.main()
