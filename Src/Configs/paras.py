"""Src/Configs/paras.py
仿真代码：Paras() 或 Paras.from_dict(...)
测试平台：Paras.from_state(...) 从测量状态 JSON 中构建
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from Src.Configs.algo_config import DEFAULT as ALGO_CFG
from Src.Configs.model_config import ModelConfig, RESNET50 as MODEL_CFG
from Src.Configs.testbed_config import DEFAULT as TESTBED_CFG

BASE_DRIVE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DRIVE / "Data"
RESULT_DIR = BASE_DRIVE / "Results"

# --- Train Data Path ---
DATA_ROOT = DATA_DIR / "CIFAR10"
WEIGHTS_DIR = MODEL_CFG.weights_dir

# --- Model Profile Path ---
MODEL_NAME = MODEL_CFG.name
RATE_CSV_PATH = MODEL_CFG.rate_csv
ACC_CSV_PATH = MODEL_CFG.acc_csv
LAYER_CSV_PATH = MODEL_CFG.layer_stats_csv

# --- Result Path ---
RESULT_TESTBED_PATH = RESULT_DIR / "Exp1_Testbed"
RESULT_SOTA_PATH = RESULT_DIR / "Exp2_Baseline"
RESULT_DYNAMIC_PATH = RESULT_DIR / "Exp3_Dynamic"
RESULT_CONVERGENCE_PATH = RESULT_DIR / "Exp4_Convergence"
RESULT_ABLATION_PATH = RESULT_DIR / "Exp5_Ablation"
RESULT_EE_MODEL_PATH = RESULT_DIR / "Exp6_EE_Model"

# --- Optimize Path ---
RESULT_GA_PATH = RESULT_DIR / "Optimize/GA"
RESULT_PPO_PATH = RESULT_DIR / "Optimize/DSCI"
RESULT_BF_PATH = RESULT_DIR / "Optimize/BF"
RESULT_TEST_PATH = RESULT_DIR / "Test"

COLORS = {
    "grey": "#999999",
    "brown": "#8D574B",
    "green": "#2ca02c",
    "purple": "#9467bd",
    "red": "#d62728",
    "blue": "#1f77b4",
}

# --- Default Parameters ---
NUM_USERS = TESTBED_CFG.num_users

NUM_LAYERS = MODEL_CFG.num_layers
EARLY_EXIT_LAYERS = list(MODEL_CFG.early_exit_layers)
NUM_EXIT_LAYERS = len(EARLY_EXIT_LAYERS)

df = pd.read_csv(LAYER_CSV_PATH)
DATA_SIZE_LAYERS = df["num_bytes"].astype(int).tolist()
COMPUTE_SIZE_LAYERS = df["approx_flops"].astype(int).tolist()

USER_FREQs = NUM_USERS * [2]
EDGE_MAX_FREQ = ALGO_CFG.edge_max_freq
CLOUD_MAX_FREQ = ALGO_CFG.cloud_max_freq

# Simulation-only wireless channel defaults. Testbed runs should pass B_u.
CHANNEL_GAINS_USERS = NUM_USERS * [2.0]
BANDWIDTH_EDGE = TESTBED_CFG.default_bw_d2e
BANDWIDTH_CLOUD = TESTBED_CFG.default_bw_e2c
BASE_STATION_POWER = 1.0
NOISE_POWER = 8e-11


@dataclass
class Paras:
    # Basic scalar parameters
    n: int = NUM_USERS
    m: int = NUM_LAYERS
    f_e_max: float = float(EDGE_MAX_FREQ)
    f_c_max: float = float(CLOUD_MAX_FREQ)
    b_e: float = float(BANDWIDTH_EDGE)
    b_c: float = float(BANDWIDTH_CLOUD)
    G: float = float(BASE_STATION_POWER)
    delta: float = float(NOISE_POWER)
    alpha: float = float(ALGO_CFG.alpha)
    beta: float = float(ALGO_CFG.beta)

    # Model and user vectors
    E: list[int] = field(default_factory=lambda: list(EARLY_EXIT_LAYERS))
    D: list[int] = field(default_factory=lambda: list(DATA_SIZE_LAYERS))
    C: list[int] = field(default_factory=lambda: list(COMPUTE_SIZE_LAYERS))
    F_u: np.ndarray = field(default_factory=lambda: np.array(USER_FREQs, dtype=float))
    H_u: np.ndarray | None = field(
        default_factory=lambda: np.array(CHANNEL_GAINS_USERS, dtype=float)
    )
    B_u: np.ndarray | None = None  # measured Device -> Edge bandwidth, Mbps
    model_cfg: ModelConfig | None = None

    rates: np.ndarray | None = field(init=False, default=None)
    accs: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        if self.model_cfg is None:
            self.model_cfg = MODEL_CFG

        self.E = list(self.E)
        self.D = list(self.D)
        self.C = list(self.C)
        self.F_u = np.asarray(self.F_u, dtype=float).reshape(-1)

        if self.H_u is not None:
            self.H_u = np.asarray(self.H_u, dtype=float).reshape(-1)
        if self.B_u is not None:
            self.B_u = np.asarray(self.B_u, dtype=float).reshape(-1)

        self._validate_runtime_shapes()

        from Src.Algorithm.Utils.parsing_data import parsing_rate_and_acc

        self.rates, self.accs = parsing_rate_and_acc(self)

    def _validate_runtime_shapes(self):
        if len(self.F_u) != self.n:
            print(f"Warning: F_u length ({len(self.F_u)}) does not match n ({self.n}).")
        if self.H_u is not None and len(self.H_u) != self.n:
            print(f"Warning: H_u length ({len(self.H_u)}) does not match n ({self.n}).")
        if self.B_u is not None and len(self.B_u) != self.n:
            print(f"Warning: B_u length ({len(self.B_u)}) does not match n ({self.n}).")
        if len(self.D) != self.m:
            print(f"Warning: D length ({len(self.D)}) does not match m ({self.m}).")
        if len(self.C) != self.m:
            print(f"Warning: C length ({len(self.C)}) does not match m ({self.m}).")

    @classmethod
    def from_dict(cls, data: dict):
        import inspect

        valid_keys = inspect.signature(cls).parameters.keys()
        filtered_params = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_params)

    @classmethod
    def from_state(cls, state: dict, model_cfg=None, algo_cfg=None):
        """Build Paras from one measured testbed state JSON.

        Args:
            state: Measured state JSON with structure:
                {
                    "users": [{"f_u": float, "BW_d2e": float}, ...],
                    "edge": {"f_e_max": float, "BW_d2e": float (optional)},
                    "cloud": {"f_c_max": float, "BW_e2c": float}
                }
            model_cfg: Model configuration (default: RESNET50)
            algo_cfg: Algorithm configuration (default: DEFAULT)

        Note:
            - ``b_e`` is NOT set here because measured path uses ``B_u`` (per-user bandwidth).
            - If ``B_u`` is provided, ``b_e`` is never used in latency computation.
            - If you need a default ``b_e`` value, it falls back to ``BANDWIDTH_EDGE`` (10.0 MHz).
        """
        model_cfg = model_cfg or MODEL_CFG
        algo_cfg = algo_cfg or ALGO_CFG

        users = state["users"]
        edge = state["edge"]
        cloud = state["cloud"]

        layer_df = pd.read_csv(model_cfg.layer_stats_csv)
        layer_bytes = layer_df["num_bytes"].astype(int).tolist()
        layer_flops = layer_df["approx_flops"].astype(int).tolist()

        return cls(
            n=len(users),
            m=int(model_cfg.num_layers),
            E=list(model_cfg.early_exit_layers),
            D=layer_bytes,
            C=layer_flops,
            f_e_max=float(edge["f_e_max"]),
            f_c_max=float(cloud["f_c_max"]),
            b_c=float(cloud["BW_e2c"]),
            alpha=float(algo_cfg.alpha),
            beta=float(algo_cfg.beta),
            F_u=np.array([u["f_u"] for u in users], dtype=float),
            H_u=None,
            B_u=np.array([u["BW_d2e"] for u in users], dtype=float),
            model_cfg=model_cfg,
        )
