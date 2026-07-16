from __future__ import annotations

import pytest

from Src.Phase2_Scheduler.Service.algo_service import AlgoService, AlgoServiceConfig


def _service(tmp_path, **overrides) -> AlgoService:
    config = AlgoServiceConfig(
        latest_solution_path=tmp_path / "latest_solution.npz",
        latest_meta_path=tmp_path / "latest_solution_meta.json",
        training_events_path=tmp_path / "training_events.jsonl",
        **overrides,
    )
    return AlgoService(config=config)


def test_force_retrain_bypasses_cache_once(tmp_path, monkeypatch):
    service = _service(tmp_path, force_retrain=True)
    cached_match = object()
    monkeypatch.setattr(
        service,
        "_best_cache_match",
        lambda signature, paras: cached_match,
    )

    first_match, first_forced = service._cache_match_for_decision({}, object())
    second_match, second_forced = service._cache_match_for_decision({}, object())

    assert first_match is None
    assert first_forced is True
    assert second_match is cached_match
    assert second_forced is False
    assert service.health()["force_retrain_pending"] is False


def test_force_retrain_requires_auto_training(tmp_path):
    with pytest.raises(ValueError, match="requires auto_train"):
        _service(tmp_path, force_retrain=True, auto_train=False)
