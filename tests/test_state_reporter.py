from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from Src.Phase3_Runtime.Shared.state_reporter import RoundClient


class RoundClientBarrierTests(unittest.TestCase):
    def test_retries_a_transient_disconnect(self):
        client = RoundClient("http://scheduler:8000", "round-1", 2)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "release_at": 100.0,
            "ready_at": {"2": 99.0},
        }

        with (
            patch(
                "Src.Phase3_Runtime.Shared.state_reporter.requests.post",
                side_effect=[requests.ConnectionError("connection closed"), response],
            ) as post,
            patch(
                "Src.Phase3_Runtime.Shared.state_reporter.time.time",
                return_value=100.0,
            ),
            patch("Src.Phase3_Runtime.Shared.state_reporter.time.sleep") as sleep,
        ):
            result = client.wait_for_request_release(7, timeout_s=5)

        self.assertEqual(result, response.json.return_value)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.2)

    def test_does_not_retry_http_errors(self):
        client = RoundClient("http://scheduler:8000", "round-1", 2)
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("409 conflict")

        with patch(
            "Src.Phase3_Runtime.Shared.state_reporter.requests.post",
            return_value=response,
        ) as post:
            with self.assertRaisesRegex(requests.HTTPError, "409 conflict"):
                client.wait_for_request_release(7, timeout_s=5)

        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
