# Exp0 Motivation Study 使用说明

本目录用于生成论文 Motivation Study 的三张图。

## 运行环境

```powershell
conda activate DSCI
```

## 运行顺序

按下面顺序运行即可生成本组实验数据和图片：

```powershell
conda run -n DSCI python Scripts\Exp0_Motivation\build_tables.py
conda run -n DSCI python Scripts\Exp0_Motivation\run_exp1_selection_effect.py
conda run -n DSCI python Scripts\Exp0_Motivation\run_exp2_coupling_failure.py
conda run -n DSCI python Scripts\Exp0_Motivation\run_exp3_decision_overhead.py
conda run -n DSCI python Scripts\Exp0_Motivation\plot_all.py
```

`build_tables.py` 会创建新的运行目录，并把 run id 写入：

```text
Scripts\Results\Exp0_Motivation\latest_run.txt
```

后续脚本默认复用最新 run。

## 各脚本作用

- `config.py`：统一配置 bundle、profile、带宽、精度约束和输出目录。
- `build_tables.py`：使用当前权重和当前数据集口径重新生成 canonical early-exit curve。
- `run_exp1_selection_effect.py`：生成 Figure 1 数据，说明阈值诱导的条件筛选效应。该实验使用每个 early-exit head 的独立条件准确率和独立早退概率，不绘制 Final Exit rate，并排除 `threshold=1.0`。
- `run_exp2_coupling_failure.py`：生成 Figure 2 数据，比较 `Local-full`、`Cloud-full`、`Split-only`、`Decoupled` 和 `Joint`。
- `run_exp3_decision_overhead.py`：生成 Figure 3 数据，比较逐请求联合决策和慢周期联合决策的调度开销。
- `plot_all.py`：读取三个实验的数据，统一输出 PDF/PNG 图片。

## 输出位置

每次运行会输出到：

```text
Scripts\Results\Exp0_Motivation\<run_id>\
  Data\
  Figures\
  Logs\
  config.json
  paper_numbers.json
```

关键文件包括：

- `Data\canonical_exit_curves.csv`：本次实验使用的 canonical early-exit 曲线。
- `Data\exp1_selection_effect.csv`：Figure 1 数据。
- `Data\exp2_coupling_failure.csv`：Figure 2 数据。
- `Data\exp3_decision_overhead.csv`：Figure 3 数据。
- `Figures\fig1a_accuracy_expectation.pdf/png`：Figure 1(a)，条件准确率与期望准确率。
- `Figures\fig1b_early_exit_probability.pdf/png`：Figure 1(b)，独立早退概率。
- `Figures\fig2_coupling_failure.pdf/png`：Figure 2。
- `Figures\fig3_decision_overhead.pdf/png`：Figure 3。
- `paper_numbers.json`：论文中可直接引用的关键数值。

## 单独运行

`build_tables.py` 是后续实验的前置步骤。它不会和历史结果比较，只会把当前权重、当前数据集 split 得到的曲线保存为本次 run 的 canonical curve。

若只想生成 Figure 3，可单独运行：

```powershell
conda run -n DSCI python Scripts\Exp0_Motivation\run_exp3_decision_overhead.py --run-id <run_id>
```

## 固定实验设置

- 模型：`resnet50-cifar10-ee-v1`
- Device profile：`device-nx1-pytorch-resnet50-cifar10`
- Edge profile：`edge-jialindesktop-pytorch-resnet50-cifar10`
- Cloud profile：`cloud-v100-pytorch-resnet50-cifar10`
- Figure 2 带宽：`B_d2e={0.5,1,2,5,10,20,50}` Mbps，`B_e2c=50` Mbps
- Figure 3 用户数：`N={1,2,4,8,16,32}`
- 网络部分：全部使用解析仿真
