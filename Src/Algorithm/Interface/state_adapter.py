"""Convert measured testbed state JSON into ``Paras``."""

from Src.Configs.algo_config import DEFAULT as DEFAULT_ALGO_CONFIG
from Src.Configs.model_config import get_model_config
from Src.Configs.paras import Paras


def to_paras(state: dict, model_cfg=None, algo_cfg=None) -> Paras:
    """Build a Paras object for one testbed decision round."""
    if model_cfg is None:
        model_cfg = get_model_config(state.get("model_name"))
    return Paras.from_state(
        state=state,
        model_cfg=model_cfg,
        algo_cfg=algo_cfg or DEFAULT_ALGO_CONFIG,
    )
