# DSCI: A Dual-Stage Collaborative Inference Framework for Device-Edge-Cloud Networks

> 本仓库包含论文 **"DSCI: A Dual-Stage Collaborative Inference Framework for Device-Edge-Cloud Networks"** 的实现代码。

该项目实现了一个面向端-边-云（Device-Edge-Cloud）网络的双阶段协同推理框架，利用多出口深度神经网络（Multi-Exit DNNs）和深度强化学习（PPO）来优化推理延迟和精度。
## 项目结构
核心代码组织如下：
```Plaintext
DSCI/
├── Data/                 # 数据集及模型中间数据 (CSV, Checkpoints)
├── Result/               # 实验结果输出目录 (日志, 图表)
├── Src/
│   ├── Experiments/      # 论文中四个实验的复现脚本
│   │   ├── Exp1_SOTA/              # 实验1：基准对比
│   │   ├── Exp2_Dynamic/           # 实验2：动态环境适应性
│   │   ├── Exp3_DSCI_Convergency/  # 实验3：算法收敛性分析
│   │   ├── Exp4_Ablation/          # 实验4：消融实验
│   │   └── Exp5_EE_Model/          # 实验5：多出口模型特性
│   ├── Objective/        # 目标函数计算 (延迟, 精度, 能耗等)
│   ├── Optimizer/        # 优化算法核心
│   │   ├── DSCI/             # PPO 算法实现 (提出的 DSCI 算法)
│   │   ├── GA/               # 遗传算法 (对比)
│   │   └── BF/               # 暴力搜索 (对比)
│   ├── Utils/            # 工具函数 (绘图, 日志, 数据解析)
│   └── paras.py          # 全局参数配置文件
└── README.md
```

## 环境依赖
安装以下 Python 库：
```Bash
pip install numpy pandas torch matplotlib seaborn
```
*本项目在 Windows 和 Linux 路径下均适配，具体路径配置见 `Src/paras.py`。*

## 核心算法复现
**核心优化算法 DSCI** 位于 `Src/Optimizer/DSCI`。运行算法即可启动 DSCI 框架的主流程，进行策略搜索与优化。
```Bash
python -m Src.Optimizer.DSCI.run_DSCI
```
该脚本将基于 `Src/paras.py` 中的配置进行训练，并将最佳策略和收敛历史保存至 `Result/Optimize/PPO`。

## 实验复现 (Experiments)
本项目提供了论文中五个主要实验的复现脚本
#### 实验 1: Baseline Comparison
#### 实验 2: Dynamic Environment
测试 DSCI 在不同网络带宽、计算资源或用户异构性下的性能
```Bash
# 资源异构性实验
python -m Src.Experiments.Exp2_Dynamic.run_resource_dynamic
# 用户异构性实验
python -m Src.Experiments.Exp2_Dynamic.run_user_dynamic
```
- **输出**: 资源变化趋势图、策略切换点分析。

#### 实验 3: Convergence Analysis
分析 DSCI (基于 PPO) 的训练收敛过程，包括总效用、熵值变化、精度-时延变化
```Bash
python -m Src.Experiments.Exp3_DSCI_Convergency.run_convergence
```
- **输出**: 效用收敛曲线、策略熵值、精度-时延变化曲线

#### 实验 4: Ablation Analysis
对比不同策略（仅终端、仅边缘、仅云端、协同推理等）下的性能表现。
```Bash
python -m Src.Experiments.Exp4_Ablation.run_ablation
```
- **输出**: 气泡图与柱状图，展示不同策略在时延与精度上的trade-off

#### 实验 5: Early-Exit Model Analysis
从头训练带有早退机制的 ResNet50 模型（基于 CIFAR-10）并分析早退与之对其的影响
```Bash
python -m Src.Experiments.Exp5_EE_Model.Resnet_Train_and_Evaluate.resnet50_train
```
- **说明**: 该脚本包含三个阶段的训练（Backbone -> Exit Branch 1 -> Exit Branch 2）。
- **输出**: 训练日志及模型权重文件。
## 参数
所有全局参数均在 `Src/paras.py` 中定义，请修改该文件来调整实验设置：
- **网络环境**: `EDGE_MAX_FREQ`, `CLOUD_MAX_FREQ`, `BANDWIDTH_EDGE` 等。
- **用户设置**: `NUM_USERS` (用户数量)。
- **模型设置**: `EARLY_EXIT_LAYERS` (早退点位置)。
- **优化权重**: `alpha` (延迟权重), `beta` (精度权重)。
```Python
class Paras:
    n: int = 10           # 用户数量
    alpha: float = 1.0    # Delay weight
    beta: float = 5.0     # Accuracy weight
    # ...
```
