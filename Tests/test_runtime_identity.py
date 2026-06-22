import unittest

from Src.Phase3_Runtime.Device.runtime_v2 import _select_user_decision
from Src.Phase3_Runtime.Shared.request_identity import request_identity


class RuntimeIdentityTests(unittest.TestCase):
    def test_selects_v2_user_decision(self):
        user = _select_user_decision({"user": {"user_id": 7}}, 7)
        self.assertEqual(user["user_id"], 7)

    def test_selects_v1_batch_decision_by_user_id(self):
        user = _select_user_decision(
            {"users": [{"user_id": 5}, {"user_id": 2}]}, 2
        )
        self.assertEqual(user["user_id"], 2)

    def test_rejects_missing_user(self):
        with self.assertRaises(ValueError):
            _select_user_decision({"users": [{"user_id": 0}]}, 3)

    def test_requires_complete_identity(self):
        with self.assertRaises(ValueError):
            request_identity({"user_id": 1})


if __name__ == "__main__":
    unittest.main()
