# SEAM 论文实验 Src 底层修改记录

日期：2026-07-13

## 1. 修改目的

本次修改不实现统一实验启动器、模型训练、论文 baseline 或绘图代码，而是补齐五类论文实验共同依赖的 Src 底层语义：

1. 同步且可约束的统一决策问题；
2. 多请求共享链路与共享 worker 的竞争模型；
3. 每个请求批次的并发执行屏障；
4. 跨 Device、Edge、Cloud 的统一请求 trace；
5. PPO、Random、GA 共用的目标函数计数与收敛 trace。

修改完成后，实验脚本只需要构造 `state` 和 `DecisionSpec`，调用同步求解接口，再消费统一的预测指标与运行时观测指标，不需要理解各优化器或三端执行器的内部日志格式。

## 2. 新增同步决策接口

新增文件：

- `Phase2_Scheduler/Service/decision_solver.py`
- `Phase2_Scheduler/Objective/evaluator.py`

公共入口：

```python
from Src.Phase2_Scheduler.Service import DecisionSpec, solve_decision

result = solve_decision(state, spec)
```

该接口是同步接口：调用返回时，本次 `state` 的优化已经结束，不会返回在线服务中的默认解、旧缓存解，也不会在后台继续训练。

### 2.1 DecisionSpec

`DecisionSpec` 支持以下约束：

| 字段 | 可选值或含义 |
| --- | --- |
| `coordination` | `joint` 或 `independent` |
| `split_rule` | `optimize` 或 `fixed` |
| `exit_rule` | `optimize`、`disabled` 或 `fixed` |
| `allowed_split_pairs` | 允许搜索的合法 `(b1, b2)` 集合 |
| `fixed_split` | 所有用户共用或逐用户指定的固定切分对 |
| `fixed_threshold` | 固定阈值标量、出口字典或逐用户出口字典 |
| `stages` | `joint` 或 `split_then_exit` |
| `optimizer` | `ppo`、`random`、`ga` 或 `static` |
| `seed` | 随机种子 |
| `evaluation_budget` | 中央目标函数评估预算 |
| `threshold_step` | Random 搜索使用的阈值离散步长 |
| `optimizer_options` | 优化器专用参数 |

消融实验可统一表达为：

| 方法 | DecisionSpec 约束 |
| --- | --- |
| Split only | `split_rule="optimize"`，`exit_rule="disabled"` |
| Early-exit only | `split_rule="fixed"`，`fixed_split=(final, final)`，`exit_rule="optimize"` |
| Independent Split+Exit | `stages="split_then_exit"` |
| SEAM | 切分与出口均为 `optimize`，`stages="joint"` |
| 单层架构 | `allowed_split_pairs=((final, final),)` |
| 端-边两层 | `allowed_split_pairs` 只放入所有满足 `b2=final` 的合法对 |
| 三层架构 | 允许 manifest 定义的全部合法切分对 |
| 静态策略 | `optimizer="static"`，切分和出口均固定 |

PPO 的动作空间和动作落地逻辑也同步接入了这些约束。固定切分时，PPO 输出的切分动作不会覆盖指定切分；禁用早退时，所有出口阈值保持为 1；指定合法切分集合时，策略网络只从该集合中选择。

### 2.2 SolveResult

同步求解统一返回：

- 原始变量 `X`、`Y`、`F_e`、`F_c`；
- 编码后的部署决策；
- 每个用户的 `expected_accuracy`、`expected_latency`、`expected_utility`；
- `utility_sum` 和 `utility_mean`；
- 统一 optimizer trace；
- 目标函数调用次数；
- `setup_s`、`solve_s`、`total_s`；
- 实际使用的 `DecisionSpec`；
- 包含 profile、带宽、共享资源和约束摘要的 SHA-256 状态签名。

`decision_source` 使用 `synchronous:<optimizer>:<coordination>`，可以直接判断结果是否来自论文实验同步接口。

### 2.3 joint 与 independent

- `joint`：全部用户进入一个共享资源目标函数，一次联合求解。
- `independent`：把 N 用户状态拆成 N 个单用户状态，分别使用对应随机种子和约束求解，再合并 `X/Y`。

`independent` 合并后会在原始多用户共享状态上额外评估一次，用于生成可比较的预测指标。这次合并评估也通过中央 evaluator 计数，并在 trace 中标记为 `merged_shared_evaluation`；它不会改变已经得到的单用户决策。

## 3. 共享资源竞争模型

新增文件：

- `Phase2_Scheduler/Objective/shared_resources.py`

修改入口：

- `Phase2_Scheduler/Objective/compute_latency.py`
- `Phase2_Scheduler/paras.py`

### 3.1 WorkerPoolResource

Edge 和 Cloud 使用固定 worker 数量的 FCFS 模型。每个任务包含到达时间和期望服务时间；当并发任务数超过 worker 数时，后续任务产生显式的 `edge_queue` 或 `cloud_queue`。

服务时间来自现有 segment profile，并根据早退概率折算为期望工作量。不会到达 Edge 或 Cloud 的请求不会虚假占用远端 worker，也不会产生远端排队时间。

### 3.2 LinkResource

D2E 和 E2C 使用工作守恒的 processor-sharing 模型：

- 同一 `link_id` 上发生重叠的传输共享链路总容量；
- 每条流仍受自身测得的最大速率限制；
- 不同 `link_id` 的传输互不竞争；
- 链路调度显式考虑任务到达时间。

共享资源状态新增以下字段：

```text
user.d2e_link_id
user.d2e_capacity_mbps       # 可选
edge.d2e_capacity_mbps       # 可选，共享 D2E 总容量
cloud.e2c_link_id
cloud.e2c_capacity_mbps
state.shared_resource_model
```

多用户调用 `solve_decision` 时默认启用 `shared_resource_model`。在线 RoundCoordinator 在用户数大于 1 时也会把该标记写入调度状态。

共享模型输出互斥的预测分项：

```text
device_compute
d2e_transfer
edge_queue
edge_compute
e2c_transfer
cloud_queue
cloud_compute
total
```

N=1 时模型退化为原 segment-profile 计算、传输和协议开销之和。

### 3.3 模型边界

该模型是基于早退概率和 segment profile 的确定性期望工作量模型，不是新的 contention profile 标定体系。它用于让 Phase 2 决策感知共享资源竞争；论文中的最终时延仍必须使用 Phase 3 的 `observed_*` 实测数据。

## 4. 请求级并发屏障

修改文件：

- `Phase2_Scheduler/Service/round_coordinator.py`
- `Phase2_Scheduler/Service/api_server.py`
- `Phase3_Runtime/Shared/state_reporter.py`
- `Phase3_Runtime/Device/run_device.py`

新增接口：

```text
POST /api/v2/rounds/{round_id}/requests/{request_seq}/ready/{user_id}
```

每个样本的执行顺序为：

1. 每台参与设备报告 `request_seq` ready；
2. Coordinator 保存每台设备的 `ready_at`；
3. 全部设备到齐后生成统一 `release_at`；
4. Device 等待到 `release_at`，然后立即开始推理；
5. Device 保存 `actual_start_at` 和 `start_skew_s`；
6. 下一批请求再次执行相同屏障。

屏障等待发生在 `run_partitioned_inference` 计时开始之前，因此不计入 `T_total`。每条请求记录新增：

```text
request_seq
barrier_ready_at_utc
barrier_release_at_utc
actual_start_at_utc
start_skew_s
```

Device 默认启用请求级屏障。`--no-request-barrier` 只用于兼容旧的非同步运行流程，不应用于正式多端并发实验。

## 5. RequestTrace 与互斥时延

新增文件：

- `Phase3_Runtime/Shared/request_trace.py`

修改文件：

- `Phase3_Runtime/Device/runtime_v2.py`
- `Phase3_Runtime/Device/run_device.py`
- `Phase3_Runtime/Edge/run_edge.py`
- `Phase3_Runtime/Cloud/run_cloud.py`
- `Phase3_Runtime/Shared/fixed_worker_pool.py`
- `Phase3_Runtime/Shared/mnn_segment_worker.py`
- `Shared/Partitioning/pytorch_executor.py`

Device、Edge、Cloud 现在追加各自的 node trace，不再覆盖上游记录。PyTorch 和 MNN 分段执行器分别记录：

- segment 计算时间；
- early-exit head 计算时间；
- softmax、置信度和阈值判断时间；
- 当前节点实际执行的 segment。

FixedWorkerPool 在任务进入进程池和开始执行之间记录 queue wait，并保存 worker 内实际执行时间。

最终 `RequestTrace` 包含：

```text
request_id / user_id / sample_id
prediction / label / correct
exit_id / exit_boundary_id / exit_location / confidence
executed_segments_by_node
device_compute
d2e_transport
edge_queue
edge_segment_compute
edge_exit_head_compute
edge_exit_check
e2c_transport
cloud_queue
cloud_segment_compute
cloud_exit_head_compute
cloud_exit_check
unattributed_overhead
total_latency
```

其中：

```text
unattributed_overhead = total_latency - 所有已知互斥分项之和
```

因此“互斥分项 + residual”必须严格闭合到端到端时延。Residual 可以包含序列化、socket 协议、Python 调用和尚未单独计时的运行时开销，不应被直接解释为网络传输。

旧 `T_*` 字段暂时保留用于兼容和排查，但汇总器改为显式字段白名单，不再遍历所有 `T_*` 后求和。D2E 别名计算也修正为扣除嵌套的 E2C roundtrip，避免跨节点时间重复计算。

## 6. 统一效用与优化 trace

请求级运行时效用统一为：

```text
observed_utility = alpha * correct - beta * observed_latency_seconds
```

默认参数来自 `Phase2_Scheduler/algo_config.py`：

```text
alpha = 1.0
beta = 5.0
```

两者仍可通过 `DSCI_OBJECTIVE_ALPHA` 和 `DSCI_OBJECTIVE_BETA` 环境变量覆盖。

指标命名规则：

- 优化器输出只使用 `expected_*`；
- Phase 3 实测只使用 `observed_*`；
- 单请求与常规实验使用请求平均 Utility；
- 扩展性实验可以先求每台设备的平均 Utility，再对设备求和；
- reward adapter 显式返回 `utility_sum` 和 `utility_mean`，不再返回含义不清的 `round_reward`。

所有优化器通过 `ObjectiveEvaluator` 调用目标函数。统一 trace schema 为：

```text
optimizer
step
objective_evaluations
current_utility
best_utility
expected_accuracy
expected_latency
elapsed_s
```

Random、GA 和 PPO 因而可以按相同的 objective evaluation budget 比较，不需要实验脚本解析各自内部日志。

## 7. 文件修改清单

### 7.1 新增文件

| 文件 | 作用 |
| --- | --- |
| `Phase2_Scheduler/Service/decision_solver.py` | 同步约束求解接口、DecisionSpec、SolveResult |
| `Phase2_Scheduler/Objective/evaluator.py` | 中央目标函数评估、计数和统一 trace |
| `Phase2_Scheduler/Objective/shared_resources.py` | 共享链路和固定 worker FCFS 模型 |
| `Phase3_Runtime/Shared/request_trace.py` | 跨节点 trace 合并和时延闭合 |
| `tests/test_paper_evaluation_core.py` | 本次底层能力的单元测试 |

### 7.2 主要修改文件

| 模块 | 修改内容 |
| --- | --- |
| `Phase2_Scheduler/paras.py` | 解析共享资源标识、容量和 worker 配置 |
| `Phase2_Scheduler/Objective/compute_latency.py` | 接入共享资源时延分解 |
| `Phase2_Scheduler/Optimizer/DSCI/agent.py` | PPO 接入 DecisionSpec 和中央 evaluator |
| `Phase2_Scheduler/Optimizer/DSCI/networks.py` | 策略网络支持受限切分对 |
| `Phase2_Scheduler/Service/round_coordinator.py` | 请求级屏障和多用户共享模型标记 |
| `Phase2_Scheduler/Service/api_server.py` | 暴露请求 ready 接口 |
| `Phase2_Scheduler/Service/reward_adapter.py` | 拆分 utility sum/mean |
| `Phase2_Scheduler/Service/algo_service.py` | 返回新的效用字段 |
| `Phase3_Runtime/Device/run_device.py` | 屏障调用、observed 指标和白名单汇总 |
| `Phase3_Runtime/Device/runtime_v2.py` | Device trace 初始化与最终闭合 |
| `Phase3_Runtime/Edge/run_edge.py` | 追加 Edge trace 并向下游传播 |
| `Phase3_Runtime/Cloud/run_cloud.py` | 追加 Cloud trace |
| `Phase3_Runtime/Shared/fixed_worker_pool.py` | 记录 queue wait 和 worker elapsed |
| `Phase3_Runtime/Shared/mnn_segment_worker.py` | MNN 分段、head 和判断计时 |
| `Shared/Partitioning/pytorch_executor.py` | PyTorch 分段、head 和判断计时 |

同时更新了 `Phase2_README.md`、`Phase3_README.md` 和 Service 公共导出。

## 8. 测试与当前验证状态

新增核心测试命令：

```powershell
conda run -n DSCI python -m unittest discover `
  -s tests `
  -p test_paper_evaluation_core.py `
  -v
```

结果：14 项测试全部通过，覆盖：

- N=1 共享模型退化；
- 超过 worker 数时产生 queue wait；
- 共享链路 processor sharing；
- 不到达远端的请求不占用远端队列；
- fixed split、disabled exit 和合法切分约束；
- 静态同步接口不返回 default/cache；
- independent 等于逐用户单独求解后合并；
- 中央 evaluation budget 和统一 trace；
- 请求屏障的共同 release 和 ready 时间；
- RequestTrace 执行路径与时延闭合；
- utility sum/mean。

另外通过：

```powershell
python -m compileall -q Src tests

$env:PYTHONPATH='.;.\tests'
conda run -n DSCI python -m unittest `
  test_six_experiments.SixExperimentTests.test_registry_and_exits `
  test_six_experiments.SixExperimentTests.test_segment_equivalence_cpu_resnet `
  -v
```

公共接口导入和 `git diff --check` 也已通过。

全量测试中的两个既有数据测试目前仍因本地缺少 CIFAR-10 train/val manifest 文件而无法执行，这不是本次 Src 修改引入的失败。

## 9. 尚未完成的验证

本次只完成代码层和本地单元测试验证，尚未完成以下实机验收：

1. 5 台 Raspberry Pi 5 的逐请求屏障与启动偏差；
2. Edge/Cloud worker 饱和时 observed queue wait；
3. 真实 Wi-Fi 共享链路下 joint 与 independent 的差异；
4. Device、Edge、Cloud 跨机器时钟和完整 RequestTrace 闭合；
5. PPO、Random、GA 在正式 bundle 和相同 evaluation budget 下的收敛对比。

正式实验前应先运行一个 `ResNet-50 + ImageNet-100、Wi-Fi、N=2、单轮` 的实机 pilot，确认请求确实同步、queue wait 为正、时延分项闭合且 `expected_*` 与 `observed_*` 没有混用，再编写和执行完整实验矩阵。

## 10. 本次明确未修改的范围

- 未新增一键实验入口或断点续跑目录；
- 未修改六个模型的训练流程；
- 未实现 DADS、Cutting-Edge、BranchyNet、CEED、I-SplitEE；
- 未新增复杂 contention profile 标定体系；
- 未实现带宽重复探测、重复轮次组织和论文绘图；
- 保留现有 AlgoService 在线默认/缓存/后台训练模式，论文实验改用独立同步接口。

