# model_loader_mnn.py
import torch
import numpy as np
import MNN
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]          # 项目根目录
MNN_MODEL_DIR = BASE_DIR / "Data" / "Weights" / "mnn"   # 存放 .mnn 文件的目录

# 阶段号到 MNN 模型文件名的映射
DEVICE_MODEL_MAP = {
    2: "device_end2.mnn",
    3: "device_end3.mnn",
    4: "device_end4.mnn",
}

# 各阶段特征的正确形状（用于将一维输出重塑为 4D 张量）
FEATURE_SHAPES = {
    2: (1, 512, 28, 28),
    3: (1, 1024, 14, 14),
    4: None,
}

EXIT_LAYER_BY_STAGE = {2: 57, 3: 103, 4: 128}


class MNNDeviceModel:
    def __init__(self, stage):
        model_path = MNN_MODEL_DIR / DEVICE_MODEL_MAP[stage]
        if not model_path.exists():
            raise FileNotFoundError(f"MNN model not found: {model_path}")
        self.interpreter = MNN.Interpreter(str(model_path))
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.stage = stage
        # 获取输出名称，用于调试
        self.output_names = list(
            self.interpreter.getSessionOutputAll(self.session).keys())
        print(
            f"[MNN] Loaded model for stage {stage}, outputs: {self.output_names}")

    def forward_partial(self, input_tensor, start, end):
        """
        输入: input_tensor (torch.Tensor or numpy) shape (1,3,224,224)
        返回: (features, logits, confidence, prediction)
        """
        # 转换为 numpy 并确保形状为 (1,3,224,224)
        if isinstance(input_tensor, torch.Tensor):
            np_img = input_tensor.cpu().numpy()
        else:
            np_img = input_tensor
        if np_img.ndim == 3:
            np_img = np.expand_dims(np_img, 0)

        # 输入 MNN
        tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,
                               np_img, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        # 获取输出字典
        outputs = self.interpreter.getSessionOutputAll(self.session)

        # 根据 stage 解析输出
        if self.stage == 2:
            # 预期输出: feature2, logits2
            feat_tensor = outputs.get("feature2")
            logits_tensor = outputs.get("logits2")
            if feat_tensor is None or logits_tensor is None:
                # 如果没有按名称找到，按索引取前两个
                values = list(outputs.values())
                feat_tensor, logits_tensor = values[0], values[1]
            feat_np = np.array(feat_tensor.getData()).copy()
            logits_np = np.array(logits_tensor.getData()).copy()
            # 重塑特征
            target_shape = FEATURE_SHAPES[2]
            if feat_np.shape[0] == np.prod(target_shape):
                feat_np = feat_np.reshape(target_shape)
            features = torch.from_numpy(feat_np)
        elif self.stage == 3:
            feat_tensor = outputs.get("feature3")
            logits_tensor = outputs.get("logits3")
            if feat_tensor is None or logits_tensor is None:
                values = list(outputs.values())
                feat_tensor, logits_tensor = values[0], values[1]
            feat_np = np.array(feat_tensor.getData()).copy()
            logits_np = np.array(logits_tensor.getData()).copy()
            target_shape = FEATURE_SHAPES[3]
            if feat_np.shape[0] == np.prod(target_shape):
                feat_np = feat_np.reshape(target_shape)
            features = torch.from_numpy(feat_np)
        else:  # stage == 4
            # 只有一个输出 logits
            logits_tensor = list(outputs.values())[0]
            logits_np = np.array(logits_tensor.getData()).copy()
            features = None

        if logits_np is None:
            return features, None, None, None

        # 计算置信度和预测
        exp_logits = np.exp(logits_np - np.max(logits_np))
        probs = exp_logits / np.sum(exp_logits)
        confidence = float(np.max(probs))
        prediction = int(np.argmax(logits_np))

        return features, logits_np, confidence, prediction


class MNNLazyLoader:
    """延迟加载器，按需加载设备模型"""

    def __init__(self):
        self.cache = {}

    def forward_partial(self, input_tensor, start, end):
        if end not in DEVICE_MODEL_MAP:
            raise ValueError(f"Unsupported end stage: {end}")
        if end not in self.cache:
            self.cache[end] = MNNDeviceModel(end)
        return self.cache[end].forward_partial(input_tensor, start, end)


_MODEL = None


def load_full_model():
    """返回一个支持 forward_partial 接口的模型对象（MNN 后端）"""
    global _MODEL
    if _MODEL is None:
        _MODEL = MNNLazyLoader()
    return _MODEL


def stage_end_from_partition_boundary(boundary, default):
    """将分区边界（层数）转换为阶段号（0-4）"""
    if boundary is None:
        return int(default)
    boundary = int(boundary)
    if boundary <= 4:
        return 0
    if boundary <= 27:
        return 1
    if boundary <= 57:
        return 2
    if boundary <= 103:
        return 3
    return 4


def threshold_for_stage(exit_thresholds, stage):
    """根据阶段号返回退出层索引和阈值"""
    exit_layer = EXIT_LAYER_BY_STAGE.get(stage)
    if exit_layer is None:
        return None, None
    if stage == 4:
        return exit_layer, None
    return exit_layer, float(exit_thresholds[str(exit_layer)])
