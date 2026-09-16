import unittest
from unittest.mock import patch

import bot


class ConnectionStatusTest(unittest.TestCase):
    def setUp(self):
        self._saved = (bot._app, bot._last_poll_error, bot._token)
        bot._last_poll_error = None
        bot._token = ""

        def restore():
            bot._app, bot._last_poll_error, bot._token = self._saved

        self.addCleanup(restore)

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


if __name__ == "__main__":
    unittest.main()
