---
status: done
Created: 2026-06-02
Updated: 2026-06-13
type: research
---
# 理论分析
## 原始模型
$$\begin{align}
& T^e_i = \sum_{j} \frac{c_j}{f^e_i} \\
& \text{s.t.} \sum_i f^e_i \leq F^e_{\max}
\end{align}
$$
- $c_j$：第 $j$ 层的 FLOPs，纯数学计数量，与硬件无关，由模型结构查表获得
- $f^e_i$：分配给任务 $i$ 的 CPU 主频（GHz），可优化变量
- $T^e_i$：用户 $i$ 在 edge 上的计算时延
- $F^e_{\max}$：设备标称的最高主频

## 问题分析
- **分子（浮点运算量）**：相同 FLOPs 的两层卷积，一个通道多空间小、一个通道少空间大，时延有差距
- **分母（CPU 运算频率）**：假设一个时钟周期恰好完成一次浮点运算（$1\ \text{Hz} = 1\ \text{FLOP/s}$），但在实际设备上
	- SIMD 指令集一个周期可同时执行 4-16 次浮点运算
	- 超标量流水线一个周期可发射多条指令
	- 对 ResNet 这类算子，内存带宽往往才是瓶颈
	- Conv+BN+ReLU 在硬件上是一个 kernel，算子融合
- **时延=分子/分母**：$T^e_i = \sum_{j} \frac{c_j}{f^e_i}$ 使用运算量除以计算速度得到的**时延估计**在真机实验上不准确。
- **结果**：==算法优化目标为时延的函数，使用的 reward 信号依赖于时延估计，导致策略收敛到次优==

## 修改思路
> 凸优化依赖 $T \propto 1/f$ 形式。需要不改变优化结构，对分子和分母重新标定含义与数值
$$T = \frac{\text{工作量}}{\text{速率}}$$
- **==真机测试==**
	- $T^{\text{meas}}_{\text{total}}$（$s$）：**目标设备上** 模型所有层的端到端推理时延
	- $T^{\text{meas}}_j$（$s$）：**目标设备上** 模型第 $j$ 层单独的推理时延
	- $c =\displaystyle\sum_j c_j$（$\text{FLOPS}$）：模型各层的理论 FLOPs $c_j$ 是已知的（与硬件无关）
- **==等效吞吐量==**
$$
\Theta^e = \frac{c}{T^{\text{meas}}_{\text{total}}} \ (\text{FLOPS/s})
$$
	- 这台设备跑这个模型，平均每秒等效处理多少次浮点运算。不是 CPU 频率，也不是理论算力，而是一个实测的**经验值**，综合当前机器的 SIMD、缓存、内存带宽、算子融合等所有真实因素
- **==等效计算量==**
$$\hat{c}_j = T^{\text{meas}}_j \times \Theta^e \ (\text{FLOPS})$$
	- 这台设备跑这个模型，第 $j$ 层的"等效计算量"，用 $\Theta^e$ 修正
- **==修正时延模型==**：
$$T^e_i = \sum_{j} \frac{\hat{c}_j}{f^e_i} = \frac{\sum_{j} T^{\text{meas}}_j \cdot \Theta^e}{f^e_i}$$
- 当单任务满载时，$f^e_i = \Theta^e$：
$$T^e_i = \sum_j \frac{\hat{c}_j}{\Theta^e} = \sum_j \frac{T^{\text{meas}}_j \cdot \Theta^e}{\Theta^e} = \sum_j T^{\text{meas}}_j = T^{\text{meas}}_{\text{total}} \quad \checkmark$$

### 当前采用方案与适用边界

当前系统保留两条明确分离的时延路径：

1. **仿真模式 `simulation_resource_mode`**：继续使用上述等效吞吐量与等效计算量模型，保留 $T\propto1/f$ 形式，用于兼容依赖连续算力变量 $F_e/F_c$ 的旧算法和理论实验。
2. **真机模式 `fixed_worker_pool`**：不再假设运行时能够执行连续算力配额，直接测量目标设备上每个可执行 Segment 的时延，并按切分区间查表求和：

$$
T_{\text{compute}}(node,b_s,b_e)
=
\sum_{k=b_s}^{b_e-1}T^{\text{profile}}_{node,k}
$$

考虑早退时，每个 Segment 按其被执行的概率加权。真机模式的计算时延不再通过 $\hat c/f$ 推导，连续资源如何执行的问题单独由 Bug 2 处理。

因此，本 Bug 的核心结论是：**理论 FLOPs 和 CPU 主频不能直接作为真机时延模型；必须先在目标设备上测量。** 等效吞吐量用于保留旧优化结构，实测 Segment 时延用于当前真实部署。

## 举例
- 假设模型只有 3 层，在树莓派上测量：

| 层 $j$  | 理论 FLOPs $c_j$ | 实测时延 $T^{\text{meas}}_j$ |
| ------ | -------------- | ------------------------ |
| 1      | 1,000          | 0.02 s                   |
| 2      | 1,000          | 0.08 s                   |
| 3      | 3,000          | 0.10 s                   |
| **合计** | **5,000**      | **0.20 s**               |

- **==原始模型==**
	层 1 和层 2 FLOPs 相同（都是 1,000），所以原始模型认为它们的计算时延完全一样。但实测层 2 比层 1 慢 4 倍（0.08 vs 0.02）——可能是因为层 2 是 memory-bound 的。
- **==校准==**
$$\Theta^e = \frac{5{,}000}{0.20} = 25{,}000 \text{ FLOP/s}$$

| 层 $j$  | 原始 $c_j$  | 实测时延 $T^{\text{meas}}_j$ | 等效 $\hat{c}_j = T^{\text{meas}}_j \times \Theta^e$ |
| ------ | --------- | ------------------------ | -------------------------------------------------- |
| 1      | 1,000     | 0.02 s                   | $0.02 \times 25{,}000 = 500$                       |
| 2      | 1,000     | 0.08 s                   | $0.08 \times 25{,}000 = 2{,}000$                   |
| 3      | 3,000     | 0.10 s                   | $0.10 \times 25{,}000 = 2{,}500$                   |
| **合计** | **5,000** | **0.20 s**               | **5,000**                                          |
- **==验证==**
	- 单任务满载（$f^e_i = \Theta^e = 25{,}000$）：
$$T = \frac{500 + 2{,}000 + 2{,}500}{25{,}000} = \frac{5{,}000}{25{,}000} = 0.20 \text{ s} = T^{\text{meas}}_{\text{total}} \quad \checkmark$$
	- 两个任务平分算力（$f^e_1 = f^e_2 = 12{,}500$），每个慢一倍。：
$$T_1 = T_2 = \frac{5{,}000}{12{,}500} = 0.40 \text{ s} = 2 \times T^{\text{meas}}_{\text{total}}$$

---

# 为什么不能用 nn-Meter

> [!important] 本质
> - **nn-Meter 解决**："在没有目标设备 $D$ 访问权限的情况下，预测模型 $M$ 在该设备上的推理时延。"因为神经网络架构搜索阶段不可能把每个候选网络都在设备上跑一遍，nn-Meter 用来做离线筛选。
> - **本文需要解决**："在设备 $D$ 上，分配资源 $f$ 时，每一层的时延是多少"（固定设备，变化资源，建模时延-资源关系）。
> 
> **问题**：
> 1. 在特定的硬件上实测代价高（之前部署的代码是开源针对已训练好的 Cortex-A76 + TFLite 2.1 实现）
> 2. 即使重新针对特定硬件训练，费劲部署后的结果也不准确（得到的仍然是一个预测时延，但是部署了测试的代码为什么不用实测的时延计算，并不是不能访问目标设备）

## 一、nn-Meter 的作用
- **设计目标**：在不访问目标设备的情况下，预测一个神经网络在某个（硬件 × 运行时）组合上的推理时延。
- **工作原理**：
	1. 将模型分解为实际执行的 kernel（考虑算子融合，Conv+BN+ReLU 合并为一个 kernel）
	2. 对每个 kernel 提取特征向量（类型、输入输出维度、stride 等）
	3. 用针对 **==目标设备预训练的 Random Forest 模型==** ，将特征向量映射到时延预测值（ms）
- **关键**：nn-Meter 的输出是某个特定设备、特定运行时、**满载单任务独占条件**下的**绝对时延**。是一个固定操作点的预测器，不是资源-时延关系的建模工具

---

## 二、本文的目标

本文需要构建一个时延模型：
$$T^e_i = \sum_{j} \frac{\hat{c}_j}{f^e_i}$$
其中 $f^e_i$ 是分配给设备 $i$ 的计算资源（可优化变量），$\hat{c}_j$ 是某种"层级计算量"。模型需要同时满足两个要求：
- **要求 A（硬件感知）**：$\hat{c}_j$ 必须反映目标硬件的真实特性（算子融合、内存带宽、SIMD 等），而不是与硬件无关的 FLOPs
- **要求 B（可伸缩）**：时延随 $f^e_i$ 的变化符合 $1/f$ 关系，凸优化能在不同资源分配下求解

---

## 三、结合方案
利用 nn-Meter 修正的方案是：用 $T^{\text{nn}}_j$（nn-Meter 预测的逐层时延）计算：
$$\Theta^e = \frac{\sum_j c_j}{T^{\text{nn}}_{\text{total}}}, \quad \hat{c}_j = T^{\text{nn}}_j \cdot \Theta^e$$
校准实际上是做：**按 nn-Meter 预测的层间时延比例，重新分配原始 FLOPs**。
$$\sum_j \hat{c}_j = \sum_j T^{\text{nn}}_j \cdot \Theta^e = T^{\text{nn}}_{\text{total}} \cdot \frac{\sum_j c_j}{T^{\text{nn}}_{\text{total}}} = \sum_j c_j$$
$$\hat{c}_j = c_j^{\text{original}} \cdot \underbrace{\frac{T^{\text{nn}}_j / T^{\text{nn}}_{\text{total}}}{c_j / \sum c_j}}_{\text{硬件特性修正因子}}$$
因此 nn-Meter 给出的是**满载单任务独占条件下的绝对时延**，是资源-时延曲线上的一个固定采样点。完全没有回答"当 $f^e_i$ 变化时，时延如何变化"这个问题。这意味着：即使 nn-Meter 对 Jetson Nano 的预测完全准确，用它推出的 $\hat{c}_j$ 代入 $T = \hat{c}_j / f^e_i$ 时，仍然隐含了"时延与资源严格成 $1/f$ 反比"这个未经验证的假设。对内存带宽受限的层（BN、大 feature map pooling），这个假设本身就是错的——即便 $f^e_i$ 减半，时延几乎不变。

---

## 四、问题分析
1. **工具与本文需求错位**。nn-Meter 解决的是固定资源配置下的时延预测问题，而本文需要建模时延随可分配资源 $f^e_i$ 变化的关系。前者是曲线上的一个点，后者是整条曲线。无论 nn-Meter 的预测精度多高，它都无法提供 $1/f$ 缩放是否在目标硬件成立的信息。
2. **部署极复杂。** nn-Meter 提供了训练自定义 predictor 的机制，理论上可以为任意设备生成专属预测器。但是 nn-Meter 的 backend 系统默认针对 Android 设备，通过 ADB 连接采集数据。Jetson Nano 运行的是嵌入式 Linux，没有 ADB，需要完全自定义 backend 适配层——包括 kernel 自动生成、部署执行、时延采集的完整 pipeline。
3. **预测而不是测量**。即使部署成功，结果也不是准确值，因为训练自定义 predictor 的过程是：在设备上跑大量 kernel 配置 → 采集时延 → 训练 RF 模型 → 用 RF 模型预测新 kernel 的时延。整个过程的本质是：先在设备上跑数据，nn-Meter 用这些数据训练了一个近似模型，然后用这个近似模型去预测。最终得到的仍然是一个带误差的预测，而不是直接测量值。
4. **代价和收益不匹配。** 训练自定义 predictor 本身就需要在设备上做大量 kernel profiling。这个工作量和直接对模型 segment 做实测 profiling 是同等量级的——但前者得到的是有误差的 RF 预测，后者得到的是精确的实测值。

因此，在拥有目标设备访问权限的前提下，先训练 predictor，再用 predictor 预测不如直接在设备上实测。费这么大力气，得到一个比直接测量更差的结果，没有理由这样做。

---

# 实验修改

> [!note]
> 以下第 1–4 点对应保留 $T\propto1/f$ 结构的 `simulation_resource_mode`。当前真机主路径采用后文“当前真机实验路径”中的实测 Segment 时延，不再执行连续 $F_e/F_c$ 配额。

1. **数据准备**：在每台目标设备（端/边/云）上跑 profiling，得到逐层时延 $T_j^{\text{meas}}$，算出该设备的 $\Theta$ 和 $\hat{c}_j$。注意设备不同，所以 $\hat{c}_j$ 各不相同（$\hat{c}_j^u$、$\hat{c}_j^e$、$\hat{c}_j^c$）
2. `paras.py`：原来一个 `C` 对三个设备通用，改为三套 `C_u`、`C_e`、`C_c`，分别加载对应设备的 $\hat{c}_j$。`f_e_max` 和 `f_c_max` 的值从标称频率改为对应设备的 $\Theta^e$、$\Theta^c$。用户端同理，`F_u` 从 CPU 频率改为 $\Theta^u$
3. `compute_latency.py`：三段时延函数分别使用对应设备的 C（local 用 `C_u`，edge 用 `C_e`，cloud 用 `C_c`）。去掉 `* 1e9` 的 GHz→Hz 转换，因为现在 $f$ 和 $\hat{c}$ 已经在同一量纲（FLOP/s 和 FLOP）下。
4. `agent.py`：`compute_iota_kappa` 里的 `c` 改为用 `C_e` 和 `C_c`（iota 用边缘的，kappa 用云端的）。`allocate_resources` 本身不需要改，因为它只用 `f_e_max` 和 `f_c_max`，这两个值已经在 paras 里换成了 $\Theta$

## 当前真机实验路径

1. 在 Device、Edge、Cloud 的真实后端和固定 Worker 配置下，对统一 Manifest 中的原子 Segment 进行 profiling。
2. Profile 记录每个 Segment 的平均时延、中位时延、P95 和校准时延，并保证：

$$
\sum_kT^{\text{calibrated}}_k=T^{\text{measured}}_{\text{total}}
$$

3. Scheduler 在 `fixed_worker_pool` 模式下直接读取 Device、Edge、Cloud 各自的 Segment profile，不再读取 CPU 标称主频或计算用户级 $F_e/F_c$。
4. 修改 Worker 数、每 Worker 线程数、推理后端或模型 Manifest 后，必须重新生成对应设备的 Segment profile。

## 验收标准

- 仿真模式下，$\sum_j\hat c_j/\Theta=T^{\text{meas}}_{\text{total}}$。
- 真机模式下，完整模型的 Segment 校准时延之和等于端到端实测时延。
- 同一 Segment 在 Device、Edge、Cloud 分别使用各自的实测 profile。
- 真机 reward 使用真实总时延，并分别统计预测计算时延与真实计算时延误差。
