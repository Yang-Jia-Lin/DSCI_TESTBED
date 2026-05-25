# DSCI 动机实验说明（Exp0_Motivation）

本文档说明 `Scripts/Exp0_Motivation/` 下两个动机实验**在代码中如何实现**、与论文论点/原实验计划的对应关系，以及如何复现。实现采用**解析期望仿真**（非真实多机推理），以便快速、可重复地验证结构性规律。

---

## 1. 动机与分析框架（论文论点）

### 1.1 客观物理规律

在端边云协同推理中，早退阈值 $\tau$ 与最优切分点 $X^*$ 并非独立变量，而是通过期望传输流量 $\mathbb{E}[V(\tau)]$ 耦合：


$$\tau \;\Rightarrow\; \text{各层退出概率} \;\Rightarrow\; \mathbb{E}[V(\tau)] \;\Rightarrow\; X^*(\tau)$$


代码中体现为：`simulator.py` 里 `transmission_ratio = 1 - Σ r_i`（对 `exit_layer ≤ split_layer` 的早退头求和），传输与边缘计算时延均乘以该比例。

### 1.2 现有工作的两类困境

| 困境 | 代表方法 | 本仓库对照实现 |
|------|----------|----------------|
| 忽视 $\tau$–$X$ 耦合 | Neurosurgeon / BranchyNet 式「单边优化」 | **EE-Only**、**SC-Only**、**Decoupled** |
| 请求粒度联合决策 | MEOCI、I-SplitEE | **Per-request 调度器**（Exp2） |

### 1.3 DSCI 的对应

| 创新点 | 动机实验 | 代码体现 |
|--------|----------|----------|
| 期望空间显式建模 $X^*(\tau)$ | 实验 1 | **DSCI-Joint** 在 $(X,\tau)$ 上网格联合搜索 |
| 控制环与推理环解耦、$O(1)$ 广播 | 实验 2 | **QuasiStaticScheduler**（DSCI） |

---

## 2. 目录与运行环境

### 2.1 目录结构

```
Scripts/Exp0_Motivation/
├── Motivation_README.md          # 本文档
├── utils/
│   ├── config.py                 # 算力、RTT、调度超参
│   ├── network_sim.py            # 传输/计算时延辅助函数
│   └── output_paths.py           # 时间戳目录与 latest.txt
├── exp1_decoupling_failure/      # 动机实验 1：解耦失效
│   ├── model_wrapper.py
│   ├── strategies.py
│   ├── simulator.py
│   ├── run_exp1.py
│   └── plot_exp1.py
└── exp2_scalability/             # 动机实验 2：可扩展性
    ├── scheduler.py
    ├── run_exp2.py
    └── plot_exp2.py
```

### 2.2 解释器（已验证）

```text
%USERPROFILE%\.conda\envs\DSCI\python.exe   # Python 3.10.19
```

依赖：`numpy`, `torch`, `pandas`, `matplotlib`, `scipy`，以及项目内 `Models/`、`Data/`、`Src/Utils/plot_utils.py`。

### 2.3 一键复现（在项目根目录 `DSCI_testbed/`）

```powershell
$py = "$env:USERPROFILE\.conda\envs\DSCI\python.exe"
cd D:\Coding\Python\DSCI_testbed

& $py Scripts/Exp0_Motivation/exp1_decoupling_failure/run_exp1.py
& $py Scripts/Exp0_Motivation/exp1_decoupling_failure/plot_exp1.py

& $py Scripts/Exp0_Motivation/exp2_scalability/run_exp2.py
& $py Scripts/Exp0_Motivation/exp2_scalability/plot_exp2.py
```

绘图脚本默认读取 `Results/Exp0_Motivation/latest.txt`；若 Exp1/Exp2 连续运行，需对 Exp1 指定时间戳：

```powershell
& $py Scripts/Exp0_Motivation/exp1_decoupling_failure/plot_exp1.py --timestamp 20260520_080309
```

### 2.4 输出约定

每次 `run_exp*.py` 创建：

```text
Results/Exp0_Motivation/YYYYMMDD_HHMMSS/
├── logs/exp1.log 或 exp2.log
├── data/exp1_results.json 或 exp2_results.json
└── figures/
    ├── exp1_main.{pdf,png}
    ├── exp1_split_drift.{pdf,png}
    ├── exp2_drift.{pdf,png}
    └── exp2_throughput_overhead.{pdf,png}
```

`latest.txt` 始终指向**最后一次**运行的目录。

---

## 3. 共享数据与模型（实验 1 基础）

### 3.1 模型

| 项目 | 实现取值 | 来源 |
|------|----------|------|
| 网络结构 | `MultiEEResNet50`（Bottleneck, blocks `[3,4,6,3]`, CIFAR-10） | `Models/ModelNet/Resnet50.py` |
| 权重 | `Models/Weights/ResNet50_multi_EE_model.pth` | 加载失败时仅用 CSV 元数据继续 |
| 早退头层索引 | **57**, **103** | `Src/Configs/model_config.py` |
| 末层 / 总层数 | **127** / **128** | 同上 |
| 早退阈值 \(\tau\) | 仅层 **57、103** 两个早退头 | CSV `exit1_rate` / `exit2_rate` |
| 候选切分点 \(X\) | **1～127 任意层** 枚举 | `0`=Local，与早退头位置无关 |

> **与原计划差异**：文档中曾写 ResNet-56/MobileNetV2；本仓库统一使用项目已有的 **ResNet-50 多早退头** 与对应 CSV，与主实验 testbed 一致。

### 3.2 CSV 数据

| 文件 | 用途 |
|------|------|
| `Data/Resnet50_rates.csv` | 阈值 $\tau$ → `exit1_rate`, `exit2_rate`（%，线性插值） |
| `Data/Resnet50_layer_stats.csv` | 逐层 `num_bytes`、`approx_flops` → 传输量与累积算力 |
| `Data/Resnet50_accs.csv` | 本动机实验未直接用于时延（主实验精度用） |

### 3.3 硬件/网络仿真参数（`utils/config.py`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `DEVICE_GFLOPS` | 0.5 | 端侧算力 |
| `EDGE_GFLOPS` | 10.0 | 边缘算力 |
| `RTT_MS` | 20.0 | 固定 RTT（与原计划 `netem` 一致） |

---

## 4. 动机实验 1：解耦失效定理

### 4.1 实验目标（对应论文）

量化验证：

1. 忽略 $\mathbb{E} [V(\tau)]$ 时切分点**系统性偏深**（相对联合最优）；
2. 带宽受限时解耦策略可能出现**性能劣于 Local** 的「倒置区」；
3. 期望空间**联合优化**（DSCI）在同一仿真器下取得更低期望时延。

### 4.2 对照策略（`strategies.py`）

| 策略 | 切分点 $X$ | 阈值 $\tau$ | 建模含义 |
|------|-------------|--------------|----------|
| **Local** | 0（无卸载） | 1.0（不参与优化） | 全本地，$ \mathbb{E}[V]=0 $ |
| **EE-Only** | 固定末层 127 | 在 `tau_grid` 内搜索 | 仅早退，不切分卸载 |
| **SC-Only** | 在 `{57,103,127}` 枚举 | 1.0（无早退） | 隐含 $\mathbb{E}[V]=V_{max}$ |
| **Decoupled** | 先按 SC-Only 得 $X^*$，再固定 $X^*$ 搜 $\tau$ | 两阶段独立 | **解耦失效**的核心对照 |
| **DSCI (Joint)** | $(X,\tau)$ 笛卡尔积网格搜索 | 联合 | 期望空间联合最优（仿真版） |

各策略在每组自变量下调用同一 `simulate_latency()`，保证对比公平。

### 4.3 自变量

| 变量 | 代码 | 说明 |
|------|------|------|
| 带宽 $B$ | `[0.5, 1, 2, 4, 8]` Mbps | 主自变量；图中 `invert_xaxis()`，右→左为恶化 |
| 数据集难度（代理） | `easy` / `hard` 两组 `tau_grid` | 见下节「简化说明」 |

```python
THRESHOLD_GROUPS = {
    "easy": [0.85, 0.88, 0.90, 0.92],   # 高 τ → 高早退率
    "hard": [0.55, 0.60, 0.65, 0.70],   # 低 τ → 低早退率
}
```

### 4.4 因变量与记录字段

| 因变量 | JSON 字段 |
|--------|-----------|
| 平均端到端时延 | `avg_latency_ms` |
| 选定切分点 | `split_layer_chosen` |
| 使用阈值 | `exit_threshold_used` |
| 实际传输比例 | `actual_transmission_ratio` |

**时延分解**（端侧 / 传输 / 边缘）在 `simulator.py` 内分三项计算后相加；当前 JSON **未单独导出**三项，若论文需要可在 `simulate_latency` 返回值中扩展。

### 4.5 期望时延公式（`simulator.py`）

设切分层为 $X$，早退头层集合 $\mathcal{E}=\{57,103\}$，对应退出率 $r_i(\tau)$（来自 CSV，0~1）：

\[
\rho(X,\tau) = \max\left(0,\; 1 - \sum_{i:\, e_i \le X} r_i(\tau)\right)
\]

\[
T_{\mathrm{e2e}} = T_{\mathrm{dev}}(X) + \rho \cdot T_{\mathrm{tx}}(X,B) + \rho \cdot T_{\mathrm{edge}}(X)
\]

- $T_{\mathrm{dev}}(X) = \dfrac{\sum_{l\le X} \mathrm{FLOPs}_l}{\mathrm{GFLOPS}_{\mathrm{dev}}}$
- $T_{\mathrm{tx}} = \dfrac{\mathrm{bytes}(X)}{B} + \mathrm{RTT}$
- $T_{\mathrm{edge}} = \dfrac{\mathrm{FLOPs}_{\mathrm{total}} - \mathrm{FLOPs}_X}{\mathrm{GFLOPS}_{\mathrm{edge}}}$

`split_layer=0` 时仅计算全模型本地时延，$\rho=0$。

### 4.6 与原实验计划的简化 / 差异

| 原计划 | 实际实现 | 原因 |
|--------|----------|------|
| CIFAR-10 易/难**子集**实测 | 用 **$\tau$ 网格** 驱动 CSV 早退率，代理易/难样本 | 已有 `Resnet50_rates.csv` 覆盖阈值–退出率关系，无需重复前向 |
| `tc netem` 实测带宽 | **解析带宽模型** + 固定 RTT | 可重复、批量扫参 |
| 1000 张实测推理时延 | **期望仿真**（无逐样本 Monte Carlo） | 动机实验聚焦结构性规律，非绝对毫秒值 |
| 时延分解堆叠图 | 未默认出图 | 可按需扩展 `simulate_latency` 返回值 |

### 4.7 图表（`plot_exp1.py`）

| 图文件 | 内容 |
|--------|------|
| `exp1_main` | 1×2 子图：easy / hard 的时延–带宽，5 策略曲线 |
| `exp1_split_drift` | hard 组：SC-Only / Decoupled / DSCI 的切分点–带宽 |

若存在 `Decoupled > Local` 的带宽点，主图标注 **Performance Inversion**（红圈 + 箭头）。

### 4.8 图表含义与动机论证

| 图 | 横轴 | 纵轴 | 证明什么 |
|----|------|------|----------|
| **exp1_main** (a)(b) | 带宽（右→左恶化） | 端到端期望时延 | **解耦失效**：Decoupled/SC-Only 相对 DSCI 的间隙；若 Decoupled 曲线高于 Local 虚线，即**性能倒置**（独立优化不如不协同） |
| **exp1_split_drift** | 带宽 | 选定切分层 \(X^*\) | **切分点漂移**：SC/Decoupled 的 \(X^*\) 随 \(B\) 与 \(\tau\) 代理组变化；DSCI 轨迹体现 \(\mathbb{E}[V(\tau)]\) 耦合下的联合选点 |

### 4.9 实测快照（20260520_082224，切分点 1～127 全枚举）

环境：`~\.conda\envs\DSCI\python.exe`，`rates_from_csv=true`，`total_flops≈1.67×10⁸`。

**现象摘要：**

- **Local** 恒约 **335 ms**（与带宽无关）。
- 切分点已在 **L1–L127** 全枚举；高带宽下 SC/Decoupled 常选 **L3**（$\mathbb{E}[V]=V_{max}$ 下偏浅切分），低带宽 easy 组约 **L58**。
- **hard @1Mbps**：DSCI **162.7 ms**（L58, $\rho=0.46$）vs Decoupled **177.1 ms**（L3, $\rho=1$）— 固定错误 $X^*$ 后调 $\tau$ 无法恢复联合最优。
- 本轮 **`inversions` 仍为空**；可降 `DEVICE_GFLOPS` 或加 0.25 Mbps 等更严带宽以触发 Decoupled>Local。

---

## 5. 动机实验 2：控制–推理耦合的可扩展性

### 5.1 实验目标

对比 **Per-request**（MEOCI / I-SplitEE 类）与 **DSCI 准静态广播** 在并发用户数 $N$ 增长时的：

1. 配置漂移（Configuration Drift）；
2. 有效推理吞吐量（req/s）；
3. 调度开销占比（%）。

### 5.2 对照实现（`scheduler.py`）

| 方案 | 类 | 决策粒度 |
|------|-----|----------|
| Per-request | `PerRequestScheduler` | 每请求触发；开销 $\propto N$ |
| DSCI 准静态 | `QuasiStaticScheduler` | 每周期一次优化 + 广播；摊销后 $\approx O(1)$ |

> **论文需注明**：未复现完整 DRL，仅用**等价调度开销解析模型**（与原计划「可用调度开销模型替代」一致）。

### 5.3 Per-request 开销模型

单轮调度总时延（ms）：

$$
T_{\mathrm{sched}} = T_{\mathrm{collect}} + N \cdot T_{\mathrm{decision}} + N \cdot T_{\mathrm{dispatch}}
$$

- $T_{\mathrm{collect}} = \dfrac{4 \cdot d_{\mathrm{state}}}{B} + \mathrm{RTT}/2$，默认 $d_{\mathrm{state}}=20$
- $T_{\mathrm{decision}} = 2$ ms/用户
- $T_{\mathrm{dispatch}} = \dfrac{16\ \mathrm{B}}{B} + \mathrm{RTT}/2$

每请求分摊：$T_{\mathrm{sched}} / N$。

**配置漂移**：在 $(X_{\mathrm{opt}},\tau_{\mathrm{opt}})=(27,0.80)$ 邻域加随机扰动，$K=20$ 请求/用户，归一化方差：

$$
\mathrm{drift} = 0.5 \frac{\mathrm{Var}(X)}{127^2} + 0.5 \frac{\mathrm{Var}(\tau)}{0.5^2}
$$

### 5.4 DSCI 准静态模型

- 周期开销：$T_{\mathrm{period}} = 500\ \mathrm{ms} + T_{\mathrm{broadcast}}$
- 周期内请求数：$N \cdot \dfrac{T_{\mathrm{period\_s}} \cdot 1000}{T_{\mathrm{infer}}}$（默认 $T_{\mathrm{infer}}=50$ ms）
- 每请求摊销：$T_{\mathrm{period}} / \text{requests}$，**drift = 0**

### 5.5 自变量与因变量

| 自变量 | `CONCURRENT_USERS = [1,2,4,8,16,32]` | \
| 因变量 | `config_drift`, `effective_throughput_rps`, `scheduling_overhead_ratio_pct` |

固定：$B=4$ Mbps，RTT=20 ms，推理时延=50 ms/请求。

### 5.6 图表（`plot_exp2.py`）

| 图文件 | 内容 |
|--------|------|
| `exp2_drift` | 漂移 vs $N$（对数横轴），填充 Per-request 与 DSCI 之间区域 |
| `exp2_throughput_overhead` | 左：吞吐量；右：调度开销柱状对比 |

### 5.6 图表含义与动机论证

| 图 | 含义 | 证明什么 |
|----|------|----------|
| **exp2_drift** | 并发用户 \(N\) vs 归一化配置漂移 | Per-request 无法维持全局一致 \((X,\tau)\)；DSCI 周期内 drift≈0，**策略稳定** |
| **exp2_throughput_overhead** 左 | 有效吞吐 vs \(N\) | DSCI 随 \(N\) 近线性扩展；Per-request 被调度开销压制 |
| **exp2_throughput_overhead** 右 | 调度开销占比柱状图 | Per-request 开销 ~20% 且随 \(N\) 显性；DSCI 摊销后 **<0.2%**，支撑 \(O(1)\) 广播论点 |

### 5.7 实测快照（20260520_080318）

| N | Per-request 开销% | 吞吐 (rps) | DSCI 开销% | 吞吐 (rps) | 倍率 |
|---|-------------------|------------|------------|------------|------|
| 1 | 30.7 | 13.9 | 1.7 | 19.7 | 1.42× |
| 16 | 20.2 | 255.3 | 0.1 | 319.7 | 1.25× |
| 32 | 19.8 | 513.2 | 0.1 | 639.7 | 1.25× |

- Per-request **drift** 维持约 **0.005**（仿真扰动较弱，曲线近似平）；DSCI **drift = 0**。
- 吞吐量：Per-request 随 $N$ 近线性扩展但斜率受开销压制；DSCI 摊销后开销极低，吞吐更高。
- 与原计划「$N\ge16$ 开销成为瓶颈」一致：Per-request 开销约 **20%** 且 drift 不收敛；DSCI 开销 **<0.2%**。

---

## 6. 论文撰写建议

1. **实验 1** 强调：在**同一期望仿真器**下，Decoupled 的 $X^*$ 与 DSCI 不一致（hard 组 0.5 Mbps）；联合搜索降低 $\rho$ 与时延。
2. **实验 2** 明确写出「调度开销等价模型，未训练 DRL」。
3. 若需严格复现「性能倒置」，可：(a) 降低 `DEVICE_GFLOPS`；(b) 增加带宽点如 0.25 Mbps；(c) 改用实测 `tc netem` 替换 `network_sim.transmission_latency_ms`。
4. 图表引用路径：`Results/Exp0_Motivation/<timestamp>/figures/`。

---

## 7. 常见问题

**Q: 为何 Exp1 中 Decoupled 与 DSCI 在 easy 组完全相同？**  
A: 在 `tau_grid=[0.85,…]` 下，第二步 $\tau$ 优化与联合搜索收敛到同一 $(X,\tau)$；hard 组低 $\tau$ 时二者分叉更明显。

**Q: 为何未跑真实 CIFAR 推理？**  
A: 动机实验验证**结构性命题**（耦合、漂移、调度复杂度），非绝对时延标定；主实验 testbed（Exp1–Exp6）负责实测。

**Q: `latest.txt` 指向错误批次？**  
A: 对绘图使用 `--timestamp YYYYMMDD_HHMMSS`。

---

*文档版本：与代码树 2026-05-20 运行结果同步；解释器 `~\.conda\envs\DSCI\python.exe`。*
