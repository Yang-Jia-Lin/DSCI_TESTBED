"""Project-wide artifact locations that do not depend on a phase."""

from dataclasses import dataclass
from pathlib import Path

from Src.Shared.Config.model_config import ModelConfig, RESNET50

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DRIVE = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "Data"
OFFLINE_TABLE_DIR = DATA_DIR / "OfflineTables"
RUNTIME_DIR = DATA_DIR / "Runtime"
SOLUTION_CACHE_DIR = RUNTIME_DIR / "SolutionCache"
DEVICE_RESULTS_DIR = RUNTIME_DIR / "DeviceResults"
RESULT_DIR = PROJECT_ROOT / "Scripts" / "Results"


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Filesystem locations derived from static model metadata."""

    model: ModelConfig

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def dataset_root(self) -> Path:
        return DATA_DIR / self.model.dataset_name

    @property
    def profile_dir(self) -> Path:
        return OFFLINE_TABLE_DIR

    @property
    def weights_dir(self) -> Path:
        return DATA_DIR / "Weights"

    @property
    def rate_csv(self) -> Path:
        return self.profile_dir / f"{self.model.artifact_prefix}_rates.csv"

    @property
    def acc_csv(self) -> Path:
        return self.profile_dir / f"{self.model.artifact_prefix}_accs.csv"

    @property
    def layer_stats_csv(self) -> Path:
        return self.profile_dir / f"{self.model.artifact_prefix}_layer_stats.csv"

    @property
    def weight_path(self) -> Path:
        return self.weights_dir / (
            f"{self.model.artifact_prefix}_{self.model.weight_kind}.pth"
        )

    @staticmethod
    def _first_existing(primary: Path, fallbacks: list[Path]) -> Path:
        if primary.exists():
            return primary
        return next((path for path in fallbacks if path.exists()), primary)

    def resolve_rate_csv(self) -> Path:
        return self._first_existing(
            self.rate_csv, [self.profile_dir / f"{self.model.name}_rates.csv"]
        )

    def resolve_acc_csv(self) -> Path:
        return self._first_existing(
            self.acc_csv, [self.profile_dir / f"{self.model.name}_accs.csv"]
        )

    def resolve_layer_stats_csv(self) -> Path:
        return self._first_existing(
            self.layer_stats_csv,
            [self.profile_dir / f"{self.model.name}_layer_stats.csv"],
        )

    def resolve_weight_path(self) -> Path:
        return self._first_existing(
            self.weight_path,
            [
                self.weights_dir / "full_model.pth",
                self.weights_dir / "ResNet50_multi_EE_model.pth",
            ],
        )


RESNET50_PATHS = ModelArtifactPaths(RESNET50)

# Compatibility constants for existing ResNet50 experiment scripts.
DATA_ROOT = RESNET50_PATHS.dataset_root
WEIGHTS_DIR = RESNET50_PATHS.weights_dir
MODEL_NAME = RESNET50.name
RATE_CSV_PATH = RESNET50_PATHS.resolve_rate_csv()
ACC_CSV_PATH = RESNET50_PATHS.resolve_acc_csv()
LAYER_CSV_PATH = RESNET50_PATHS.resolve_layer_stats_csv()

RESULT_TESTBED_PATH = RESULT_DIR / "Exp1_Testbed"
RESULT_SOTA_PATH = RESULT_DIR / "Exp2_Baseline"
RESULT_DYNAMIC_PATH = RESULT_DIR / "Exp3_Dynamic"
RESULT_CONVERGENCE_PATH = RESULT_DIR / "Exp4_DSCI_Convergency"
RESULT_DSCI_CONVERGENCY_PATH = RESULT_CONVERGENCE_PATH
RESULT_ABLATION_PATH = RESULT_DIR / "Exp5_Ablation"
RESULT_EE_MODEL_PATH = RESULT_DIR / "Exp6_EE_Model"

RESULT_GA_PATH = RESULT_DIR / "Optimize/GA"
RESULT_PPO_PATH = RESULT_DIR / "Optimize/DSCI"
RESULT_BF_PATH = RESULT_DIR / "Optimize/BF"
RESULT_TEST_PATH = RESULT_DIR / "Test"
