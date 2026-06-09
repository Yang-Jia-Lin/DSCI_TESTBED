"""Validated input data for one Phase 2 scheduling problem.

``Paras`` is intentionally owned by Phase 2 rather than ``Shared``. It adapts
runtime state and offline tables into the exact arrays consumed by the
objective and optimizers.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from Src.Phase2_Scheduler.algo_config import DEFAULT as ALGO_CFG
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import RESNET50 as MODEL_CFG
from Src.Shared.Config.model_config import ModelConfig
from Src.Shared.Config.paths import RESNET50_PATHS as MODEL_PATHS
from Src.Shared.Config.paths import ModelArtifactPaths
from Src.Shared.Profiles.compute_profile import load_compute_profile

# --- Default Parameters ---
NUM_USERS = TESTBED_CFG.num_users

NUM_LAYERS = MODEL_CFG.num_layers
EARLY_EXIT_LAYERS = list(MODEL_CFG.early_exit_layers)
NUM_EXIT_LAYERS = len(EARLY_EXIT_LAYERS)


def _read_layer_stats_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(col).strip() for col in df.columns]
    required = {"num_bytes", "approx_flops"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


df = _read_layer_stats_csv(MODEL_PATHS.resolve_layer_stats_csv())
DATA_SIZE_LAYERS = df["num_bytes"].astype(int).tolist()
COMPUTE_SIZE_LAYERS = df["approx_flops"].astype(int).tolist()


USER_FREQs = NUM_USERS * [2e9]
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
    # Deprecated theoretical-FLOPs alias kept for existing simulation callers.
    C: list[float] = field(
        default_factory=lambda: [float(x) for x in COMPUTE_SIZE_LAYERS]
    )
    C_theoretical: list[float] | None = None
    C_u: np.ndarray | None = None
    C_e: np.ndarray | None = None
    C_c: np.ndarray | None = None
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
        if self.C_theoretical is None:
            self.C_theoretical = list(self.C)
        self.C_theoretical = [float(value) for value in self.C_theoretical]
        self.C = list(self.C_theoretical)
        self.F_u = np.asarray(self.F_u, dtype=float).reshape(-1)
        if self.C_u is None:
            self.C_u = np.tile(np.asarray(self.C_theoretical, dtype=float), (self.n, 1))
        else:
            self.C_u = np.asarray(self.C_u, dtype=float)
        self.C_e = np.asarray(
            self.C_theoretical if self.C_e is None else self.C_e, dtype=float
        ).reshape(-1)
        self.C_c = np.asarray(
            self.C_theoretical if self.C_c is None else self.C_c, dtype=float
        ).reshape(-1)

        if self.H_u is not None:
            self.H_u = np.asarray(self.H_u, dtype=float).reshape(-1)
        if self.B_u is not None:
            self.B_u = np.asarray(self.B_u, dtype=float).reshape(-1)

        assert self.C_theoretical is not None
        assert self.C_u is not None
        assert self.C_e is not None
        assert self.C_c is not None

        self._validate_runtime_shapes()

        from Src.Phase2_Scheduler.Utils.parsing_data import parsing_rate_and_acc

        self.rates, self.accs = parsing_rate_and_acc(self)

    def _validate_runtime_shapes(self):
        assert self.C_theoretical is not None
        assert self.C_u is not None
        assert self.C_e is not None
        assert self.C_c is not None

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
        if len(self.C_theoretical) != self.m:
            print(
                f"Warning: C_theoretical length ({len(self.C_theoretical)}) "
                f"does not match m ({self.m})."
            )
        if self.C_u.shape != (self.n, self.m):
            raise ValueError(f"C_u shape {self.C_u.shape} != ({self.n}, {self.m})")
        if self.C_e.shape != (self.m,):
            raise ValueError(f"C_e shape {self.C_e.shape} != ({self.m},)")
        if self.C_c.shape != (self.m,):
            raise ValueError(f"C_c shape {self.C_c.shape} != ({self.m},)")

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
                    "users": [{
                        "f_u": float, "BW_d2e": float, "compute_profile_id": str
                    }, ...],
                    "edge": {"f_e_max": float, "compute_profile_id": str},
                    "cloud": {
                        "f_c_max": float, "BW_e2c": float, "compute_profile_id": str
                    }
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

        def positive_float(value, fallback, name, minimum=1e-6):
            try:
                out = float(value)
            except (TypeError, ValueError):
                out = float("nan")
            if not np.isfinite(out) or out <= 0:
                safe = max(float(fallback), float(minimum))
                print(
                    f"Warning: invalid measured state {name}={value!r}; "
                    f"using fallback {safe}."
                )
                return safe
            return max(out, float(minimum))

        bw_d2e_values = [
            positive_float(
                u.get("BW_d2e"),
                TESTBED_CFG.default_bw_d2e,
                f"users[{i}].BW_d2e",
                minimum=0.1,
            )
            for i, u in enumerate(users)
        ]
        bw_e2c = positive_float(
            cloud.get("BW_e2c"),
            TESTBED_CFG.default_bw_e2c,
            "cloud.BW_e2c",
            minimum=0.1,
        )

        model_paths = ModelArtifactPaths(model_cfg)
        layer_df = _read_layer_stats_csv(model_paths.resolve_layer_stats_csv())
        layer_bytes = layer_df["num_bytes"].astype(int).tolist()
        layer_flops = layer_df["approx_flops"].astype(float).tolist()
        layer_names = layer_df["layer"].astype(str).tolist()

        def load_state_profile(owner: dict, capacity_key: str, owner_name: str):
            profile_id = owner.get("compute_profile_id")
            if not profile_id:
                raise KeyError(f"{owner_name}.compute_profile_id")
            profile = load_compute_profile(
                str(profile_id),
                expected_layers=layer_names,
                expected_theoretical_flops=layer_flops,
                expected_model=model_cfg.name,
            )
            capacity = positive_float(
                owner.get(capacity_key),
                profile.theta,
                f"{owner_name}.{capacity_key}",
                minimum=1e-3,
            )
            if not np.isclose(capacity, profile.theta, rtol=1e-6, atol=1e-3):
                raise ValueError(
                    f"{owner_name}.{capacity_key}={capacity} does not match "
                    f"profile theta={profile.theta}"
                )
            return profile

        user_profiles = [
            load_state_profile(user, "f_u", f"users[{i}]")
            for i, user in enumerate(users)
        ]
        edge_profile = load_state_profile(edge, "f_e_max", "edge")
        cloud_profile = load_state_profile(cloud, "f_c_max", "cloud")

        return cls(
            n=len(users),
            m=int(model_cfg.num_layers),
            E=list(model_cfg.early_exit_layers),
            D=layer_bytes,
            C=layer_flops,
            C_theoretical=layer_flops,
            C_u=np.stack([profile.equivalent_flops for profile in user_profiles]),
            C_e=edge_profile.equivalent_flops,
            C_c=cloud_profile.equivalent_flops,
            f_e_max=edge_profile.theta,
            f_c_max=cloud_profile.theta,
            b_c=bw_e2c,
            alpha=float(algo_cfg.alpha),
            beta=float(algo_cfg.beta),
            F_u=np.array([profile.theta for profile in user_profiles], dtype=float),
            H_u=None,
            B_u=np.array(bw_d2e_values, dtype=float),
            model_cfg=model_cfg,
        )
