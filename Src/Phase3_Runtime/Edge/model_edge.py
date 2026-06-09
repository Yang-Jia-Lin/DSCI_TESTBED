import torch.nn as nn

from Src.Phase3_Runtime.Shared.model_loader import load_full_model


class EdgeModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.model = load_full_model()

    def forward(self, x):
        features, logits, conf, pred = self.model.forward_partial(x, 3, 3)
        return features
