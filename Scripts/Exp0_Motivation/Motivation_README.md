# Exp0 Motivation Study 使用说明

本目录用于生成论文 Motivation Study 的三张图。

## 运行环境

```powershell
conda activate DSCI
```

## 运行顺序

按下面顺序运行即可生成本组实验数据和图片：

```powershell
conda run -n DSCI python Scripts\Exp0_Motivation\run\build_tables.py
conda run -n DSCI python Scripts\Exp0_Motivation\run\run_exp1_selection_effect.py
conda run -n DSCI python Scripts\Exp0_Motivation\run\run_exp2_coupling_failure.py
conda run -n DSCI python Scripts\Exp0_Motivation\plot\plot_all.py
```

各脚本直接读写本实验目录下的 `result_data` 和 `result_figure`。

## 各脚本作用

- `config.py`：统一配置 bundle、profile、带宽、精度约束和输出目录。
- `build_tables.py`：使用当前权重和当前数据集口径重新生成 canonical early-exit curve。
- `run_exp1_selection_effect.py`：生成 Figure 1 数据，说明阈值诱导的条件筛选效应。该实验使用每个 early-exit head 的独立条件准确率和独立早退概率，不绘制 Final Exit rate，并排除 `threshold=1.0`。
- `run_exp2_coupling_failure.py`：生成 Figure 2 数据，在相同精度约束下比较 `Local-full`、`Cloud-full`、`Split-only`、`EE-only`、`Decoupled` 和 `Joint` 的期望时延。该实验使用 Figure 1 的 sequential early-exit flow 口径。
- `plot_all.py`：读取两个实验的数据，统一输出 Figure 1(a)、Figure 1(b) 和 Figure 2 的 PDF/PNG 图片。

## 输出位置

实验数据和图片分别输出到：

```text
Scripts\Exp0_Motivation\
  result_data\
  result_figure\
```

关键文件包括：

- `result_data\canonical_exit_curves.csv`：本次实验使用的 canonical early-exit 曲线。
- `result_data\exp1_selection_effect.csv`：Figure 1 数据。
- `result_data\exp2_coupling_failure.csv`：Figure 2 数据。
- `result_figure\fig1a_accuracy_expectation.pdf/png`：Figure 1(a)，条件准确率与期望准确率。
- `result_figure\fig1b_early_exit_probability.pdf/png`：Figure 1(b)，独立早退概率。
- `result_figure\fig2_coupling_failure.pdf/png`：Figure 2，split-threshold 耦合下的期望时延。
- `result_data\paper_numbers.json`：论文中可直接引用的关键数值。

## 单独运行

`build_tables.py` 是后续实验的前置步骤。它不会和历史结果比较，只会把当前权重、当前数据集 split 得到的曲线保存为当前实验的 canonical curve。

## 固定实验设置

- 模型：`resnet50-cifar10-ee-v1`
- Device profile：`device-nx1-pytorch-resnet50-cifar10`
- Edge profile：`edge-jialindesktop-pytorch-resnet50-cifar10`
- Cloud profile：`cloud-v100-pytorch-resnet50-cifar10`
- Figure 2 带宽：`B_d2e={60,70,80,84,86,88,89,90,91,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,125,130,140,150}` Mbps，`B_e2c=50` Mbps。该范围聚焦 no-exit split 选择从本地执行切换到 edge 执行的敏感区，并在切换点附近加密采样，用于观察阈值诱导样本流变化对最优切分的影响。
- 网络部分：全部使用解析仿真
