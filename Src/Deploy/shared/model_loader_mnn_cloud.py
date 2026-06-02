# model_loader_mnn_cloud.py
import torch
import numpy as np
import MNN
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MNN_MODEL_DIR = BASE_DIR / "Data" / "Weights" / "mnn"

# 云端模型使用 edge_start3_end4.mnn（从 stage3 特征到最终输出）
CLOUD_MODEL_PATH = MNN_MODEL_DIR / "edge_start3_end4.mnn"


class MNNCloudModel:
    def __init__(self):
        if not CLOUD_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MNN cloud model not found: {CLOUD_MODEL_PATH}")
        self.interpreter = MNN.Interpreter(str(CLOUD_MODEL_PATH))
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        print(f"[MNN Cloud] Loaded model from {CLOUD_MODEL_PATH}")

    def forward(self, input_feat):
        """
        输入: stage3 特征 (1,1024,14,14)
        返回: (logits, confidence, prediction)
        """
        if isinstance(input_feat, torch.Tensor):
            np_input = input_feat.cpu().numpy()
        else:
            np_input = input_feat
        if np_input.ndim == 3:
            np_input = np.expand_dims(np_input, 0)

        tmp_input = MNN.Tensor(np_input.shape, MNN.Halide_Type_Float,
                               np_input, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        outputs = self.interpreter.getSessionOutputAll(self.session)
        logits_tensor = list(outputs.values())[0]
        logits_np = np.array(logits_tensor.getData()).copy()

        exp_logits = np.exp(logits_np - np.max(logits_np))
        probs = exp_logits / np.sum(exp_logits)
        confidence = float(np.max(probs))
        prediction = int(np.argmax(logits_np))

        return logits_np, confidence, prediction


_cloud_model = None


def load_cloud_model():
    global _cloud_model
    if _cloud_model is None:
        _cloud_model = MNNCloudModel()
    return _cloud_model
