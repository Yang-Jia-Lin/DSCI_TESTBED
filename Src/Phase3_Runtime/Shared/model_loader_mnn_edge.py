# model_loader_mnn_edge.py
import torch
import numpy as np
import MNN  # type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MNN_MODEL_DIR = BASE_DIR / "Data" / "Weights" / "mnn"

# 边缘子模型映射 (start_stage, end_stage) -> 文件名
EDGE_MODEL_MAP = {
    (2, 2): "edge_in1_out2.mnn",  # 输入 stage1 特征 -> stage2 特征+logits2
    (2, 3): "edge_in2_out3.mnn",  # 输入 stage2 特征 -> stage3 特征+logits3
    # 输入 stage2 特征 -> 最终 logits (dummy feature)
    (2, 4): "edge_in2_out4.mnn",
    # 输入 stage2 特征 -> 仅 logits3 (dummy feature)
    (3, 3): "edge_in2_out3_logitsonly.mnn",
    # 输入 stage3 特征 -> 最终 logits (dummy feature)
    (3, 4): "edge_in3_out4.mnn",
    (4, 4): "edge_in3_out4.mnn",  # 复用，因为输入 stage3 特征到最终 logits
}


class MNNEdgeModel:
    def __init__(self, start_stage, end_stage):
        key = (start_stage, end_stage)
        if key not in EDGE_MODEL_MAP:
            raise ValueError(f"Unsupported edge stage range: {key}")
        model_path = MNN_MODEL_DIR / EDGE_MODEL_MAP[key]
        if not model_path.exists():
            raise FileNotFoundError(f"MNN edge model not found: {model_path}")
        self.interpreter = MNN.Interpreter(str(model_path))
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.start = start_stage
        self.end = end_stage
        # 获取输出名称
        self.output_names = list(
            self.interpreter.getSessionOutputAll(self.session).keys()
        )
        print(
            f"[MNN Edge] Loaded model for start={start_stage}, end={end_stage}, outputs={self.output_names}"
        )

    def forward(self, input_feat):
        """
        输入: input_feat (torch.Tensor or numpy) 形状取决于 start_stage
        返回: (output_feat, logits, confidence, prediction)
        """
        # 转换为 numpy
        if isinstance(input_feat, torch.Tensor):
            np_input = input_feat.cpu().numpy().astype(np.float32)
        else:
            np_input = np.array(input_feat, dtype=np.float32)
        if np_input.ndim == 3:
            np_input = np.expand_dims(np_input, 0)

        np_input = np.ascontiguousarray(np_input)

        # 调试信息
        print(f"[MNN Edge] Input shape: {np_input.shape}, dtype: {np_input.dtype}")

        # 输入 MNN
        input_shape = np_input.shape  # 例如 (1,512,28,28) 或 (1,1024,14,14)
        tmp_input = MNN.Tensor(
            input_shape, MNN.Halide_Type_Float, np_input, MNN.Tensor_DimensionType_Caffe
        )
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        outputs = self.interpreter.getSessionOutputAll(self.session)

        # 根据输出名称解析
        if self.end == 3:
            # 输出: feature3, logits3
            feat_tensor = outputs.get("feature3")
            logits_tensor = outputs.get("logits3")
            if feat_tensor is None or logits_tensor is None:
                values = list(outputs.values())
                feat_tensor, logits_tensor = values[0], values[1]
            feat_np = np.array(feat_tensor.getData()).copy()
            # 重塑为正确的形状 (1,1024,14,14)
            if len(feat_np.shape) == 1:
                feat_np = feat_np.reshape(1, 1024, 14, 14)
            out_feat = torch.from_numpy(feat_np)
            # 检查特征是否为 dummy（形状极小且接近零）
            if (
                out_feat is not None
                and out_feat.numel() <= 4
                and torch.all(torch.abs(out_feat) < 1e-6)
            ):
                out_feat = None
            logits_np = np.array(logits_tensor.getData()).copy()
        elif self.end == 4:
            # 输出只有 logits_final
            logits_tensor = list(outputs.values())[0]
            out_feat = None
            logits_np = np.array(logits_tensor.getData()).copy()
        else:
            out_feat = None
            logits_np = None

        if logits_np is None:
            return out_feat, None, None, None

        exp_logits = np.exp(logits_np - np.max(logits_np))
        probs = exp_logits / np.sum(exp_logits)
        confidence = float(np.max(probs))
        prediction = int(np.argmax(logits_np))

        return out_feat, logits_np, confidence, prediction


# 全局缓存
_edge_model_cache = {}


def load_edge_model(start_stage, end_stage):
    key = (start_stage, end_stage)
    if key not in _edge_model_cache:
        _edge_model_cache[key] = MNNEdgeModel(start_stage, end_stage)
    return _edge_model_cache[key]
