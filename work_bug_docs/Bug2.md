---
status: done
Created: 2026-06-05
Updated: 2026-06-13
type: research
---
# 理论分析

### 原始模型
原始优化器将 Edge 和 Cloud 计算能力表示为可连续划分的资源：
$$
\sum_i f_i^e\leq F_{\max}^e,\qquad
\sum_i f_i^c\leq F_{\max}^c
$$
对用户 $i$，计算时延满足：
$$
T_i^e=\frac{W_i^e}{f_i^e},\qquad
T_i^c=\frac{W_i^c}{f_i^c}
$$
隐含的执行假设是：多个用户同时运行，每个用户持续获得自己的算力份额。

### 问题分析
1. 上述假设对应的物理过程是**连续算力平分**，但真实 CPU 推理的执行粒度是**离散 Worker 排队**：固定数量的 Worker 进程，每次只执行一个请求，其余等待。两者描述的不是同一个物理过程。
2. 原始代码用 `torch.set_num_threads()` 将配额换算为线程数来模拟算力分配，但这只修改进程级全局配置，不隔离单个请求的资源。多请求并发时线程相互竞争，动态修改引入抖动，内存带宽受限算子也不随线程数成比例加速。
3. 因此，即使优化器求出数学上可行的 $f_i^e$，运行时也无法保证执行遵守该分配。算法预测时延基于连续配额假设，真机测量基于实际排队结果，两者没有统一的执行基础，比对无意义。

### 修改思路
保留两种执行模式：**仿真模式**沿用连续 $F_e/F_c$ 和 Bug 1 的等效吞吐量模型，用于理论验证；**真实部署模式**放弃连续算力配额，改用固定 Worker Pool 描述计算能力，资源竞争统一用 FCFS 排队建模，节点时延拆分为 $T_{\text{node}} = T_{\text{queue}} + T_{\text{compute}}$，Scheduler 用离散事件模拟器预测排队时延。真实部署模式下，优化器只优化切分边界和早退阈值，不再产生算力配额变量。

固定 Worker Pool 中，若请求 $i$ 到达时刻为 $a_i$、对应服务时间为 $s_i$，则请求被最早空闲 Worker 接收：

$$
t_i^{start}=\max(a_i,t_{\min}^{free})
$$

$$
T_i^{queue}=t_i^{start}-a_i,\qquad
t_i^{finish}=t_i^{start}+s_i
$$

其中服务时间 $s_i$ 由 Bug 1 的实测 Segment profile 和 Bug 3 的实际切分区间确定。资源分配问题因此被重新表述为：**固定并行服务能力下的请求排队问题**，而不是连续算力平分问题。

---

# 实验修改
### 修改步骤
1. **确定 Worker 配置。** 在每台 Edge/Cloud 上确定可同时执行的 Worker 数和每个 Worker 的固定线程数，保证不超过逻辑 CPU 总数。
2. **按 Worker 配置重新生成 Profile。** Segment profile 必须在与部署一致的 Worker 配置下测量，Worker 数或线程数变动后需重新 profiling。
3. **固定 Worker 线程数，禁止运行时动态修改。** 线程数在 Worker 启动时写死，请求执行期间不随用户或决策变化。
4. **从 Decision 中删除算力配额字段。** 真实部署模式下 Scheduler 只返回切分边界和早退阈值，Runtime 拒绝读取旧的 `edge_compute_alloc` / `cloud_compute_alloc` 字段。
5. **记录排队与计算时延。** Edge 和 Cloud 分别记录 `T_queue` 和 `T_compute`，供后续时延模型验证使用。
6. **验证并发行为。** 分别以 1、`worker_count`、`2 × worker_count` 个并发请求测试，确认排队时延在超载时可测量、Worker 不发生过度订阅。

### 当前实现状态
- 固定进程 Worker Pool、有界队列、固定每 Worker 线程数：已实现。
- 真实部署模式取消连续配额下发：已实现。
- FCFS 排队预测：已实现。
- 多个独立 Device 自动组成联合优化批次：未实现，属于 Bug 4。

### 使用说明

1. 真实部署必须选择 `fixed_worker_pool`，旧连续算力实验显式选择 `simulation_resource_mode`。
2. 在 Edge 和 Cloud 上确定 `worker_count`、`threads_per_worker` 和 `max_queue_size`。
3. 保证：

$$
\text{worker\_count}\times\text{threads\_per\_worker}
\leq\text{逻辑 CPU 数}
$$

4. 按实际 Worker 配置生成 Segment profile；运行时配置与 profile 不一致时拒绝启动或生成决策。
5. 真机 Decision 只包含切分边界和早退阈值，不应包含用户级算力配额。

### 与其他 Bug 的边界

- “单个 Segment 在设备上需要多长时间”属于 Bug 1。
- “模型可以在哪里切分、需要传输什么”属于 Bug 3。
- “多个独立 Device 如何组成一个联合优化批次”属于 Bug 4。
- 本 Bug 只负责 Edge/Cloud 收到多个请求后，如何以真实可执行方式共享计算能力。

### 验收标准

- 运行期间不根据请求动态修改线程数。
- 并发请求数不超过 Worker 数时可以并行执行。
- 并发请求数超过 Worker 数时产生可测量的 FCFS 排队时延。
- 分别记录 `T_queue_edge`、`T_compute_edge`、`T_queue_cloud`、`T_compute_cloud`。
- 真机 Decision 中不包含 `edge_compute_alloc`、`cloud_compute_alloc` 和 compute quota。
- 并发数为 1、`worker_count`、`2 × worker_count` 时，无明显 CPU 过度订阅。
