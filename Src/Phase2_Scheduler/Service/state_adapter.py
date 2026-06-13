"""Convert a strict bundle-scoped state JSON into Paras."""

from Src.Phase2_Scheduler.paras import Paras


def to_paras(state: dict, algo_cfg=None) -> Paras:
    return Paras.from_state(state, algo_cfg=algo_cfg)
