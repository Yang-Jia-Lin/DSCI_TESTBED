"""Static model configuration"""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    name: str = "Resnet50"
    num_layers: int = 128
    early_exit_layers: list[int] = field(default_factory=lambda: [57, 103])
    data_dir: Path = BASE_DIR / "Data"
    profile_dir: Path = BASE_DIR / "Data" / "OfflineTables"
    weights_dir: Path = BASE_DIR / "Data" / "Weights"

    @property
    def rate_csv(self) -> Path:
        return self.profile_dir / f"{self.name}_rates.csv"

    @property
    def acc_csv(self) -> Path:
        return self.profile_dir / f"{self.name}_accs.csv"

    @property
    def layer_stats_csv(self) -> Path:
        return self.profile_dir / f"{self.name}_layer_stats.csv"


RESNET50 = ModelConfig()

# Placeholder layer topology until {name}_layer_stats.csv exist for each model.
RESNET18 = ModelConfig(
    name="Resnet18",
    num_layers=128,
    early_exit_layers=[57, 103],
)

ALEXNET = ModelConfig(
    name="Alexnet",
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
