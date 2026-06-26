"""Project and model-bundle artifact locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Src.Shared.Config.model_config import ModelBundleSpec, get_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "Data"
DATASET_DIR = DATA_DIR / "Datasets"
BUNDLE_DIR = DATA_DIR / "Bundles"
PROFILE_DIR = DATA_DIR / "Profiles"
COMPUTE_PROFILE_DIR = PROFILE_DIR / "Compute"
SEGMENT_PROFILE_DIR = PROFILE_DIR / "Segments"
ARCHIVE_DIR = DATA_DIR / "Archive"
RUNTIME_DIR = DATA_DIR / "Runtime"
SOLUTION_CACHE_DIR = RUNTIME_DIR / "SolutionCache"
DEVICE_RESULTS_DIR = RUNTIME_DIR / "DeviceResults"
RESULT_DIR = PROJECT_ROOT / "Scripts" / "Results"


@dataclass(frozen=True)
class BundleArtifactPaths:
    bundle: ModelBundleSpec

    @property
    def root(self) -> Path:
        return BUNDLE_DIR / self.bundle.bundle_id

    @property
    def weight_path(self) -> Path:
        return self.root / "weights.pth"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def offline_table_path(self) -> Path:
        return self.root / "exit_curves.csv"

    @property
    def layer_stats_path(self) -> Path:
        return self.root / "layer_stats.csv"

    @property
    def dataset_root(self) -> Path:
        directory = {
            "cifar10": "CIFAR10",
            "imagenet100": "ImageNet100",
        }[self.bundle.dataset_id]
        return DATASET_DIR / directory

    @property
    def test_package_root(self) -> Path:
        return self.dataset_root / "TestSets"

    @property
    def analysis_root(self) -> Path:
        return self.root / "analysis"

    @property
    def mnn_root(self) -> Path:
        return self.root / "mnn_segments"


def bundle_paths(bundle_id: str | None = None) -> BundleArtifactPaths:
    return BundleArtifactPaths(get_bundle(bundle_id))


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
