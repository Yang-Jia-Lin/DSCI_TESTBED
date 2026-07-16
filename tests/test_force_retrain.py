from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from Src.Phase2_Scheduler.Service.algo_service import AlgoService, AlgoServiceConfig


def _service(tmp_path: Path, **overrides) -> AlgoService:
    config = AlgoServiceConfig(
        latest_solution_path=tmp_path / "latest_solution.npz",
        latest_meta_path=tmp_path / "latest_solution_meta.json",
        training_events_path=tmp_path / "training_events.jsonl",
        **overrides,
    )
    return AlgoService(config=config)


class ForceRetrainTests(unittest.TestCase):
    def test_force_retrain_bypasses_cache_once(self):
        with TemporaryDirectory() as directory:
            service = _service(Path(directory), force_retrain=True)
            cached_match = object()
            with patch.object(
                service, "_best_cache_match", return_value=cached_match
            ):
                first_match, first_forced = service._cache_match_for_decision(
                    {}, object()
                )
                second_match, second_forced = service._cache_match_for_decision(
                    {}, object()
                )

            self.assertIsNone(first_match)
            self.assertTrue(first_forced)
            self.assertIs(second_match, cached_match)
            self.assertFalse(second_forced)
            self.assertFalse(service.health()["force_retrain_pending"])

    def test_force_retrain_requires_auto_training(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires auto_train"):
                _service(
                    Path(directory), force_retrain=True, auto_train=False
                )


if __name__ == "__main__":
    unittest.main()
