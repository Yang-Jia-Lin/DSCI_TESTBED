"""Static model metadata and model-scoped artifact naming."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "Resnet50"
    model_slug: str = "resnet50"
    dataset_name: str = "CIFAR10"
    dataset_slug: str = "cifar10"
    weight_kind: str = "multi_ee"
    num_layers: int = 128
    early_exit_layers: list[int] = field(default_factory=lambda: [57, 103])

    @property
    def artifact_prefix(self) -> str:
        return f"{self.model_slug}_{self.dataset_slug}"


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
