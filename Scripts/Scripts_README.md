# Scripts README

`Scripts/` 保存论文实验、消融、收敛性分析和绘图脚本。`Src/` 负责系统实现与真机运行，`Scripts/` 负责在已有模型包、profile、调度解或运行结果基础上生成实验表格和图。

## 目录导航

| 目录 | 说明 | 主要输出 |
| --- | --- | --- |
| `EvaluationCommon/` | 实验复用的配置、artifact 检查、solution 读取、overhead 计算 | 被各实验脚本导入 |
| `Exp0_Motivation/` | motivation 和 scalability 相关实验 | `Scripts/Results/Exp0_Motivation/` |
| `Exp1_Baseline/` | overall performance 和 baseline 对比准备 | `Scripts/Results/Exp1_Baseline/` |
| `Exp2_Ablation/` | 消融实验 | `Scripts/Results/Exp2_Ablation/` |
| `Exp3_Convergency_and_Overhead/` | PPO 收敛性、优化器对比、Phase overhead | `Scripts/Results/Exp3_Convergency_and_Overhead/` |
| `Results/` | 脚本生成或整理后的实验结果 | CSV、JSONL、PNG、PDF |

## 运行前准备

实验脚本通常依赖以下产物：

- `Data/Bundles/<bundle_id>/manifest.json` 和 `exit_curves.csv`。
- `Data/Profiles/<profile_id>/`。
- `Data/Runtime/SolutionCache/latest_solution.npz` 和 `latest_solution_meta.json`，或指定的历史 solution。
- Phase 3 真机运行生成的测量日志或 runtime summary。

如果这些产物不存在，先按 [Src_README](../Src/Src_README.md) 和 [Phase1_README](../Src/Phase1_Offline/Phase1_README.md) 准备系统。

## 常用命令

Exp1 overall performance：

```powershell
python -m Scripts.Exp1_Baseline.run_SOTA_baseline
```

Exp2 ablation：

```powershell
python -m Scripts.Exp2_Ablation.run_ablation
python -m Scripts.Exp2_Ablation.plot_ablation
```

Exp3 convergence and overhead：

```powershell
python -m Scripts.Exp3_Convergency_and_Overhead.run_convergence
python -m Scripts.Exp3_Convergency_and_Overhead.plot_convergency
```

Exp0 scalability：

```powershell
python -m Scripts.Exp0_Motivation.exp2_scalability.run_exp2
python -m Scripts.Exp0_Motivation.exp2_scalability.plot_exp2
```

具体参数以各脚本的 `--help` 为准，例如：

```powershell
python -m Scripts.Exp1_Baseline.run_SOTA_baseline --help
```

## 结果目录

脚本默认把结果写入 `Scripts/Results/` 下对应实验目录。常见文件包括：

| 文件类型 | 说明 |
| --- | --- |
| `.csv` | 表格结果、消融结果、overhead 汇总 |
| `.json` / `.jsonl` | 实验配置、逐轮指标、baseline 指标 |
| `.png` / `.pdf` | 论文图或分析图 |
| `latest.txt` | 指向最近一次实验结果目录 |

`Results/` 中的历史结果可用于对照，但复现时应优先确认输入 solution、bundle 和 profile 是否与目标实验一致。

## 与 Src 的关系

- Phase 1 生成模型包、manifest、exit curves 和 profile。
- Phase 2 生成调度 solution 和在线缓存。
- Phase 3 生成真机运行测量结果。
- `Scripts/` 读取这些产物，生成论文实验表格、图和对照结果。

相关文档：

- [DATA_README](../Data/DATA_README.md)
- [Src_README](../Src/Src_README.md)
- [Phase2_README](../Src/Phase2_Scheduler/Phase2_README.md)
