"""Static model configuration"""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    name: str = "Resnet50"
    model_slug: str = "resnet50"
    dataset_name: str = "CIFAR10"
    dataset_slug: str = "cifar10"
    weight_kind: str = "multi_ee"
    num_layers: int = 128
    early_exit_layers: list[int] = field(default_factory=lambda: [57, 103])
    data_dir: Path = BASE_DIR / "Data"
    profile_dir: Path = BASE_DIR / "Data" / "OfflineTables"
    weights_dir: Path = BASE_DIR / "Data" / "Weights"

    @property
    def artifact_prefix(self) -> str:
        return f"{self.model_slug}_{self.dataset_slug}"

    @property
    def rate_csv(self) -> Path:
        return self.profile_dir / f"{self.artifact_prefix}_rates.csv"

    @property
    def acc_csv(self) -> Path:
        return self.profile_dir / f"{self.artifact_prefix}_accs.csv"

    @property
    def layer_stats_csv(self) -> Path:
        return self.profile_dir / f"{self.artifact_prefix}_layer_stats.csv"

    @property
    def weight_path(self) -> Path:
        return self.weights_dir / f"{self.artifact_prefix}_{self.weight_kind}.pth"

    @property
    def legacy_rate_csv(self) -> Path:
        return self.profile_dir / f"{self.name}_rates.csv"

    @property
    def legacy_acc_csv(self) -> Path:
        return self.profile_dir / f"{self.name}_accs.csv"

    @property
    def legacy_layer_stats_csv(self) -> Path:
        return self.profile_dir / f"{self.name}_layer_stats.csv"

    @property
    def legacy_weight_paths(self) -> list[Path]:
        return [
            self.weights_dir / "full_model.pth",
            self.weights_dir / "ResNet50_multi_EE_model.pth",
        ]

    @staticmethod
    def _first_existing(primary: Path, fallbacks: list[Path]) -> Path:
        if primary.exists():
            return primary
        for path in fallbacks:
            if path.exists():
                return path
        return primary

    def resolve_rate_csv(self) -> Path:
        return self._first_existing(self.rate_csv, [self.legacy_rate_csv])

    def resolve_acc_csv(self) -> Path:
        return self._first_existing(self.acc_csv, [self.legacy_acc_csv])

    def resolve_layer_stats_csv(self) -> Path:
        return self._first_existing(self.layer_stats_csv, [self.legacy_layer_stats_csv])

    def resolve_weight_path(self) -> Path:
        return self._first_existing(self.weight_path, self.legacy_weight_paths)


RESNET50 = ModelConfig()

# Placeholder layer topology until {name}_layer_stats.csv exist for each model.
RESNET18 = ModelConfig(
    name="Resnet18",
    model_slug="resnet18",
    num_layers=128,
    early_exit_layers=[57, 103],
)

ALEXNET = ModelConfig(
    name="Alexnet",
    model_slug="alexnet",
    num_layers=128,
    early_exit_layers=[57, 103],
)

MODEL_REGISTRY: dict[str, ModelConfig] = {
    cfg.name: cfg for cfg in (RESNET50, RESNET18, ALEXNET)
}


def get_model_config(
    model_name: str | None, *, default: ModelConfig | None = None
) -> ModelConfig:
    """Resolve ``ModelConfig`` from a testbed ``model_name`` string."""
    if not model_name:
        return default or RESNET50
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model_name {model_name!r}. Known models: {known}"
        ) from exc
