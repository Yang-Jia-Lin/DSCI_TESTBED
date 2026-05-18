"""Src/paras.py

全局路径与实验常量
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DRIVE = Path(__file__).resolve().parents[1]

# 2. 基于根目录定义子路径 (pathlib 会自动处理 / 和 \)
DATA_DIR = BASE_DRIVE / "Data"
RESULT_DIR = BASE_DRIVE / "Results"

# --- Train Data Path ---
DATA_ROOT = DATA_DIR / "CIFAR10"
WEIGHTS_DIR = BASE_DRIVE / "Models" / "Weights"

# --- Simulate Path ---
MODEL_NAME = "Resnet50"
RATE_CSV_PATH = DATA_DIR / f"{MODEL_NAME}_rates.csv"
ACC_CSV_PATH = DATA_DIR / f"{MODEL_NAME}_accs.csv"
LAYER_CSV_PATH = DATA_DIR / f"{MODEL_NAME}_layer_stats.csv"

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

# User
NUM_USERS = 10

# Model
NUM_LAYERS = 128
EARLY_EXIT_LAYERS = [57, 103]  # 1 9 12 18 9 1
NUM_EXIT_LAYERS = len(EARLY_EXIT_LAYERS)
csv_path = DATA_DIR / f"{MODEL_NAME}_layer_stats.csv"
df = pd.read_csv(csv_path)
DATA_SIZE_LAYERS = df["num_bytes"].astype(int).tolist()
COMPUTE_SIZE_LAYERS = df["approx_flops"].astype(int).tolist()

# Compute
USER_FREQs = NUM_USERS * [2]
EDGE_MAX_FREQ = 20.0
CLOUD_MAX_FREQ = 50.0

# Communicate
CHANNEL_GAINS_USERS = NUM_USERS * [2.0]  # 用户的信道增益
BANDWIDTH_EDGE = 10.0
BANDWIDTH_CLOUD = 50.0
BASE_STATION_POWER = 1.0  # 基站的发射功率 W
NOISE_POWER = 8e-11  # 高斯噪声 W

COLORS = {
    "grey": "#999999",
    "brown": "#8D574B",
    "green": "#2ca02c",
    "purple": "#9467bd",
    "red": "#d62728",
    "blue": "#1f77b4",
}


@dataclass
class Paras:
    # 基础类型
    n: int = NUM_USERS  # 终端用户数量
    m: int = NUM_LAYERS  # DNN模型层数
    f_e_max: float = float(EDGE_MAX_FREQ)  # 边缘服务器最大频率
    f_c_max: float = float(CLOUD_MAX_FREQ)  # 云服务器最大频率
    b_e: float = float(BANDWIDTH_EDGE)  # 边缘服务器的带宽
    b_c: float = float(BANDWIDTH_CLOUD)  # 云服务器的带宽
    G: float = float(BASE_STATION_POWER)  # 基站的发射功率
    delta: float = float(NOISE_POWER)  # 噪声功率
    alpha: float = 1.0  # delay 所占权重
    beta: float = 5.0  # accuracy 所占权重

    # 可变类型
    E: list = field(default_factory=lambda: list(EARLY_EXIT_LAYERS))  # 早退层的集合
    D: list = field(
        default_factory=lambda: list(DATA_SIZE_LAYERS)
    )  # 各层的输出数据大小
    C: list = field(default_factory=lambda: list(COMPUTE_SIZE_LAYERS))  # 各层的计算大小
    F_u: np.ndarray = field(
        default_factory=lambda: np.array(USER_FREQs)
    )  # 每个用户的处理频率
    H_u: np.ndarray = field(
        default_factory=lambda: np.array(CHANNEL_GAINS_USERS)
    )  # 每个用户的信道增益
    rates: np.ndarray | None = field(init=False, default=None)
    accs: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        self.F_u = np.asarray(self.F_u)
        self.H_u = np.asarray(self.H_u)
        # 检查用户数和配置数组长度是否匹配
        if len(self.F_u) != self.n:
            print(f"Warning: F_u length ({len(self.F_u)}) does not match n ({self.n}).")
        from Src.Algorithm.Utils.parsing_data import parsing_rate_and_acc

        self.rates, self.accs = parsing_rate_and_acc(self)

    @classmethod
    def from_dict(cls, data: dict):
        import inspect

        valid_keys = inspect.signature(cls).parameters.keys()
        filtered_params = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_params)
