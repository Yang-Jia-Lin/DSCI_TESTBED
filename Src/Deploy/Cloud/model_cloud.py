import torch.nn as nn

from Src.Deploy.shared.model_loader import load_full_model


class CloudModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.model = load_full_model()

    def forward(self, x):
        features, logits, conf, pred = self.model.forward_partial(x, 4, 4)
        return features
