---
status: doing
Created: 2026-06-05
Updated: 2026-06-13
type: research
---
# 理论分析

### 原始模型
优化器使用长度为 128 的逻辑层空间选择两个切分点：

$$
x_i=(s_{i,1},s_{i,2}),\qquad 0\leq s_{i,1}<s_{i,2}<128
$$

理论上模型被划分为：

```text
Device: [0, s1)
Edge:   [s1, s2)
Cloud:  [s2, end)
```

真实运行时为了简化，只实现了 5 个 Stage，并将任意逻辑层切分点映射为 Stage 结束位置。传输量使用单个逻辑层输出的 `num_bytes` 估计。

### 问题分析
1. 优化器认为 128 个切分位置各有不同的计算和传输代价，但运行时将多个位置映射到同一 Stage，导致多个不同的优化动作在实际执行时完全等价。Reward 信号对应的不是真正执行的切分方案，算法可能收敛到运行时无法复现的结果。
2. ResNet 包含残差连接，残差块内部的任意逻辑层不一定是合法的计算图边界。若在残差块内部切分，后续计算同时依赖主分支和 identity 分支，仅传输当前层输出会丢失后续仍需使用的张量。切分点必须由计算图依赖关系决定，而不是层编号。
3. 真实传输 payload 除主张量外还包含张量名称、元数据、序列化开销和协议固定开销，用单个逻辑层输出大小估计传输量会系统性低估。
4. 优化器选择的切分边界必须同时在 PyTorch（Edge/Cloud 执行）和 MNN（Device 执行）上可用。若 PyTorch 支持某个边界而 MNN 没有对应子图，Scheduler 仍可能下发运行时无法执行的决策。

### 修改思路
> 用 `torch.fx` 对模型建立统一计算图 Manifest，筛选 PyTorch 和 MNN 共同支持的公共边界作为合法切分位置（当前 ResNet50 共 20 个，覆盖 Stem、每个 Bottleneck 结束、池化、分类器和早退点）。相邻边界之间定义一个原子 Segment，运行时顺序执行对应区间的 Segment，保证优化器选择的边界与实际执行范围完全一致。每个边界通过 FX liveness 分析记录后续仍需使用的全部活跃张量，传输量改用实际 pickle 序列化字节数加协议固定开销估计。原 128 层索引仅保留用于论文展示和早退概率计算，不再控制真实执行。

1. 建立统一计算图 Manifest。使用 `torch.fx` 生成唯一分区 Manifest。`boundary_id` 成为优化器和运行时共同使用的切分标识。
2. 使用 FX Liveness 确定传输 Tensor Bundle。每个边界记录所有后续仍需使用的活跃张量。当前首版公共边界均位于残差块结束位置，liveness 验证表明每个边界只有一个 `main` 张量。协议仍保留多张量能力，以支持后续增加残差块内部边界。
3. 使用原子 Segment 执行任意合法区间。Manifest 相邻边界之间定义一个原子 Segment。节点接收 `start_boundary`、`end_boundary` 和命名 tensor bundle，然后顺序执行：

```text
segment[start_boundary]
...
segment[end_boundary - 1]
```

因此任意合法双切分实际执行为：

```text
Device: [0, b1)
Edge:   [b1, b2)
Cloud:  [b2, final_boundary)
```

4. 使用实际序列化字节估计传输时延。Manifest 记录每个边界 tensor bundle 的实际 pickle 序列化字节数不再使用单个逻辑层输出大小近似残差边界传输量：
$$
T_{\text{tx}}(b)=
\frac{8D_{\text{serialized}}(b)}{B}
+T_{\text{protocol}}
$$
5. PyTorch 与 MNN 共用公共边界。PyTorch 使用 Segment Executor 执行 Manifest 区间。MNN 按相同 Manifest 导出每个原子 Segment 的 ONNX/MNN 模型。启动时必须校验全部 Segment 和早退头存在；缺失时拒绝启动，不允许静默回退到旧 Stage。

---

# 实验修改

### 修改步骤

1. **生成 Manifest。** 对模型建立 FX 图，筛选 PyTorch/MNN 共同支持的公共边界，记录边界 ID、原子 Segment、早退点和每个边界的活跃张量集合。
2. **实现 PyTorch Segment Executor。** 将每个原子 Segment 建立为可独立执行的计算单元，输入输出统一为命名 tensor bundle，支持任意合法边界区间的组合执行。
3. **限制优化器动作空间。** DSCI、GA、BF 只能选择 Manifest 中的合法边界 ID，Scheduler 在收到决策请求时校验边界合法性，拒绝非法边界。
4. **修改通信 Payload。** Device、Edge、Cloud 之间传输 `manifest_id`、`boundary_id` 和完整 tensor bundle，Runtime 启动时校验本地 Manifest 与 Scheduler 版本一致。
5. **修改传输时延估计。** 使用每个边界 tensor bundle 的实际序列化字节数和实测协议固定开销，不再使用逻辑层单输出大小近似。
6. **导出 MNN Segment 并校验完整性。** 将原子 Segment 导出为 ONNX/MNN 模型，Device 端启动时检查全部 Segment 和早退头均存在，缺失时拒绝启动，不允许静默回退旧 5-Stage 逻辑。
7. **验证数值一致性。** 对任意合法双切分，比较 Device→Edge→Cloud 分段执行与完整本地模型的最终 logits 和早退结果。

### 当前实现状态
- 20 个公共边界、19 个原子 Segment Manifest：已实现。
- FX liveness 分析和实际序列化字节记录：已实现。
- PyTorch 任意合法边界区间执行：已实现。
- DSCI/GA/BF 合法边界限制、Decision/Runtime Manifest 校验：已实现。
- ONNX Segment 导出：已验证。
- MNN 数值一致性（目标设备）：待验证。
- 残差块内部多张量公共边界：协议已支持，当前公共边界集合尚未加入。

### 使用说明

1. 模型结构或公共边界规则变化后，重新生成唯一 Manifest。
2. Device、Edge、Cloud 和 Scheduler 必须使用相同 `manifest_id`。
3. 优化器和固定切分测试只能使用 Manifest 中存在的 `boundary_id`，不能继续使用任意逻辑层编号控制真实执行。
4. Runtime 接收边界后必须校验合法性，不允许映射或回退为旧 5-Stage 执行。
5. 传输时延必须使用边界完整 tensor bundle 的实际序列化字节数，并单独加入实测协议固定开销。

### 与其他 Bug 的边界

- Segment 的实测计算时延属于 Bug 1。
- Edge/Cloud 如何并发执行 Segment 属于 Bug 2。
- 多个独立 Device 如何被 Scheduler 联合优化属于 Bug 4。
- 本 Bug 只负责保证优化切分、真实执行区间和实际传输 payload 三者一致。

### 验收标准

- 当前 ResNet50 Manifest 包含 20 个公共边界和 19 个原子 Segment。
- 任意合法双切分组合的最终 logits 与完整本地模型一致。
- 非法边界在 Scheduler 和 Runtime 两端均被拒绝。
- 每个边界的传输量来自完整 tensor bundle 的实际序列化字节。
- PyTorch 与 MNN 使用完全相同的 Manifest 边界和输入输出名称。
- MNN 缺少任一 Segment 或早退头时拒绝启动，不允许静默回退旧 Stage。
