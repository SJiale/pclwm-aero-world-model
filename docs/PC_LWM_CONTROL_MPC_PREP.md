# PC-LWM MPC 控制验证与在线适应：前置决策清单（仅规划）

> 当前不实现、不训练、不运行新控制实验。本文件用于锁定下一阶段工程决策，避免在控制动作语义不清时产生错误验证结论。

## 1. 控制层级必须先选

| 维度 | S：上层监督控制 | F：直接燃油/执行机构控制 |
|---|---|---|
| Agent/MPC 动作 | 允许的 PLA/转速参考/监督指令修改 | `Wf_cmd`、作动器参考或经定义的等价动作 |
| 原 EEC | 仍执行闭环控制 | 明确旁路/协调/保护逻辑 |
| 当前以 `PLA,Alt,MN` 为输入的数据 | **可能**适用，但仍需验证参考动作覆盖 | **不适用**，须重新采集动作条件数据 |
| 控制结论 | 监督控制性能 | 直接发动机执行量控制性能 |
| 主要风险 | 参考指令与真实执行映射、EEC 限制器饱和 | 动作注入、约束、安全回退和未覆盖行为 |

**首选讨论顺序**：先评估 S 是否足以形成有价值的工程任务；若科研目标明确要求燃油级控制，再选择 F 并接受额外数据/安全工作量。不能将 S 的实验写成 F 的成果。

## 2. 需在实现前冻结的控制接口契约

建议将以下字段记录到项目配置（只做规格说明，当前不写控制代码）：

```yaml
control_level: undecided  # S or F
action:
  name: TBD
  physical_meaning: TBD
  unit: TBD
  injected_at: TBD
  bound_min: TBD  # need engineering source
  bound_max: TBD
  slew_rate_limit: TBD
  held_during_step: TBD
observation:
  channels: TBD
  sampling_interval_s: TBD
  measurement_delay_s: TBD
engine:
  integration_step_s: TBD
  existing_eec_mode: TBD
  reset_condition: TBD
  failure_fallback: TBD
constraints:
  n_limit: TBD
  temperature_limit: TBD
  fuel_limit: TBD
  termination: TBD
```

**三重语义一致性**：训练数据动作 `a_t` = MPC 优化动作 `a_t` = Simulink 真正执行的动作（含时刻/保持约定）。在同一时间步约定下验证 `observe_t` 与 `action_t` 的因果关系。

**仿真环境原型（后续）**：`reset(initial_condition) → observation`，`step(action) → observation,reward,cost,done,info`；必须记录动作实际执行值、限幅标志、闭环控制器状态、真实发动机输出、物理约束标志以及每步墙钟耗时。

## 3. 最小 MPC 设计与公平对照

初版任务：一个可行工作点附近的转速参考阶跃跟踪。成本函数候选：

\[
J = \sum_{j=1}^{H}\big(w_N(\hat N_{t+j}-N^{ref}_{t+j})^2 + w_u\lVert\Delta u_{t+j-1}\rVert^2 + w_c C_{t+j}\big).
\]

`C` 是风险/约束代理，**不能单独作为硬安全保证**。必须有动作硬限幅、动作速率限制、温度/转速监控、可行域检查、终止与控制回退逻辑。约束具体数值须来自模型参数/技术文档/既定安全规定。

| 组别 | 方法 | 关键意义 |
|---|---|---|
| C0 | 原 PI/EEC（或与控制层级匹配的传统基准） | 控制性能及约束基准 |
| C1 | GRU 或普通 RSSM + 同一 MPC 优化器 | 排除“只是 MPC 有用” |
| C2 | PC-RSSM/PC-LWM + 同一 MPC 优化器 | 检验物理锚定的实际控制增益 |
| C3 可选 | 经验证有效的 DT-PC-RSSM + 同一 MPC | 检验双尺度的增益 |
| C4 可选 | 高保真 Simulink 内环优化 | 测算最优性/计算预算，可能不满足实时要求 |

统一：动作、观测、参考轨迹、MPC 时域、优化器、算力/候选次数预算、重规划频率、种子/工况和限制。报告模型推演速度与质量，并计入数据生成和训练成本。

**核心指标**：跟踪误差及累计代价、超调与调节时间、最大温度/转速、约束违反次数/幅度、闭环失败率、单步优化耗时、控制周期达标率、训练/仿真整体成本。

## 4. 在线适应实验（在可行 MPC 后再做）

**失配场景**：选择经模型核实的 `FMV.Gain`、`FMV.Tau`、`Burn.Eff` 扰动；分突然和缓慢改变两种模式；扰动幅度不自行预设。

**区分**：

- **在线后验状态更新**：以新观测更新 RSSM 的信念状态 `h,z`，**权重不变**。
- **在线模型参数校正**：根据新交互数据更新潜在参数、残差网络或部分权重，**模型发生更新**。

在同一扰动、交互步数和控制约束下比较：固定 PC-LWM+MPC、仅后验更新+MPC、有限数据模型校正+MPC，必要时加普通 RSSM 的同样校正方案。

报告：扰动后损失增长、恢复所需交互数据/时间、累计代价、最大约束违反、更新耗时、名义工况遗忘。**预测损失下降不自动等于闭环控制性能恢复**。

## 5. Actor–Critic（可选扩展）

只有动作接口和基础 MPC 验证成立、且算力与数据预算允许时再考虑：在 PC-LWM 想象轨迹中训练 Actor–Critic，迁移至 Simulink 评估。可用 MPC 示范行为克隆初始化，所有 RL 对照（直接 Simulink、普通 RSSM、PC-LWM）必须采用公平初始化与交互预算。重点报告高保真仿真交互样本效率，而不是仅比最终回报。

**当前待解决的四个关键问题**：动作接口 S/F、约束来源、Simulink step/reset 的可控性与耗时、控制导向数据覆盖。
