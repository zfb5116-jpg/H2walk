# 当前版本：h2_stairs_dense_v3

针对“平地能走、第一阶前停住”的训练结果，在 v2 上增加有限探索预算：

| 新项 | 每一级的最大奖励 | 触发条件 |
| --- | ---: | --- |
| stair_approach | 2 | 躯干从阶沿前 65 cm 前进到前 15 cm |
| stair_lift | 3 | 对侧脚支撑，摆动脚最低脚底抬到本级高度 + 4 cm |
| stair_reach | 4 | 抬脚同时向踏面前伸 |
| stair_completion | 6 | 既有顺序接触判据确认完成新一级 |

前三项只奖励每回合每一级的历史最佳增量；退后再前进、反复抬脚不会重复得分。
这些增量先除以控制 dt，再统一乘 dt，因此表内数值是事件总奖励。
只在有前进指令、未摔倒、上楼且非平地时启用。原始踏面顺序、交替脚和接触约束继续保留。

速度跟踪乘以实际前进速度/指令速度（限制在 0 到 1），正向指令下停住为零；
no_progress 从停住 1 秒后 -2/秒，改为 0.4 秒后 -6/秒。该两项改变同时适用于平地和下楼。
课程仍依据成功率升级，不按时间强制升高台阶。

36 项测试通过，并完成 256 transitions 的 PPO 保存冒烟检查，以及 v2 策略迁移后追加 128 transitions 的检查；未完成长训练，不能宣称已学会上 20 cm 楼梯。

以下为 v2 历史设计说明，表格权重以 rewards.py 的 v3 为准。

---

# WM-LOCO 楼梯奖励映射

来源：[WM-LOCO v1，第 2.2 节](https://arxiv.org/html/2609.02542v1#S2.SS2)、
[附录 B 表 3](https://arxiv.org/html/2609.02542v1#A2)。使用 **Stairs** 列，不混入 Stones 的参数。

## 总奖励

`reward = control_dt * sum(weight[name] * raw[name])`。
权重来自表 3；论文没有给出所有标准项的完整公式/带宽，下面明确列出本仓库的 H2 选择，
不能把这些选择称为论文原式。论文三项楼梯公式按第 2.2 节实现，足部采样用简化几何近似。

| 代码项 | 权重 | 原始值/门控 |
|---|---:|---|
| linear_velocity_tracking | +3 | exp(-局部 XY 速度误差平方和 / 0.25)，同时约束侧向速度 |
| angular_velocity_tracking | +3 | exp(-局部 yaw 角速度误差平方 / 0.25) |
| heading_error | -1 | 骨盆 yaw 与累计指令目标航向的 wrap 误差平方 |
| no_progress | -2 | 前进指令 >0.05，连续 1 s 实际 world-X 前进速度低于指令的 10% |
| alive | +0.5 | 未触发失稳 |
| stand_pose_anchor | +1.5 | 零指令下 exp(-15 关节站姿误差平方和 / 0.25) |
| stand_pose_l2 | -1 | 零指令下站姿误差平方和 |
| stand_feet_contact | +0.4 | 零指令下双脚支撑比例 |
| walking_action_rate | -0.005 | 非零指令下 15 维裁剪动作相邻差平方和 |
| stair_boundary | -0.05 | 两脚体积采样进入侧边禁止区的点数 |
| stair_riser | -0.5 | 两脚体积采样进入立面 cuboid 的点数；排除顶部 4 cm |
| stair_tread | +0.5 | 按 Eq.4 的顺序、交替落脚得分 |
| feet_air_time | +0.5 | 首次重新支撑时 clip(腾空时间-0.25,0,0.35)；实际身体前进 >0.05 |
| feet_slide | -0.4 | 支撑脚世界水平速度平方和 |
| feet_flat_orientation | -0.4 | 支撑脚局部 z 轴的世界 xy 分量平方和 |
| feet_on_plane | -0.1 | 支撑脚底采样 z 与该采样位置真值地面 z 的均方误差 |
| feet_close_xy | +0.4 | 行走时 exp(-(脚间前后差平方+(横向间距-0.21)^2)/0.09) |
| hip_deviation | -0.5 | 双髋 roll/yaw 相对站姿偏差平方，排除抬腿需要的 hip pitch |
| angular_velocity_xy | -0.05 | 骨盆局部 roll/pitch 角速度平方和 |
| flat_orientation | -3 | 躯干投影重力 xy 平方和 |
| pelvis_orientation | -3 | 骨盆投影重力 xy 平方和 |
| joint_torque | -1.5e-7 | 15 关节末子步实际施加力矩平方和 |
| joint_acceleration | -1.25e-7 | 相邻控制步关节速度差/dt 的平方和 |
| joint_velocity | -1e-4 | 15 关节速度平方和 |
| mechanical_energy | -5e-5 | sum(abs(力矩*速度)) |
| joint_position_limit | -1 | 超出 MJCF 区间中间 90% 的关节角度超限量之和 |
| joint_velocity_limit | -1 | 超出标称速度上限 90% 的超限量之和 |
| torque_limit | -0.01 | 超出 PD 力矩上限 90% 的超限量之和 |
| undesired_contact | -1 | >5 N 的非脚地面接触或身体自碰撞涉及的身体数 |

零指令定义为指令向量范数 <=0.05；普通训练只采样正前进指令，站立项可通过 reset options 的零指令测试，
默认前进训练中不会支付站立奖励。标称速度、力矩与 PD 参数是简化 H2 模型的初始设置，
不是从论文 G1 的电机参数直接移植。

## 顺序踏面逻辑

- 当前区域足底接触比例至少 0.5，且该脚存在 >5 N 向上地形接触，才打开窗口。
- `tau = clip(当前区域深度 / (2*command_vx*dt), 5, 50)`；末端平台使用其真实深度。
- 期望脚得 `rho`，错误脚得 `0.3*rho`，另一脚重叠扣 `0.7*max(rho_other-0.05,0)`。
- 维持同一只脚至少 `ceil(0.3*tau)` 帧，才进入下一块踏面；不能跳过中间区域拿奖励。
- 一个区域已消费后不再奖励；窗口到期不重开，但允许晚到的持续接触消费区域，避免永久堵塞。
  这是论文未明确的恢复细节所采用的 H2 实现选择。
- 接触序列按行进方向 world-X 排列，上楼踏面高度逐级升高，下楼逐级降低。
- 体积每碰撞 geom 5x3x3=45 点，足底 7x5=35 点；数量影响几何惩罚的尺度。
  这与论文基于 mesh 的传感器不是同一套采样。
- 边界采用台阶侧边的 y=±0.60 m 中央通道、x=0.70..4.3 m；这是自带 4 m 宽场景的适配，
  不声称等于论文每地形 tile 的具体边界布局。

## 删除与主动腰部的例外

删除旧的 raw-forward、upright 加分、固定高度 swing_clearance、low_clearance、landing、
stumble、stall、action L2、成功 +10、摔倒 -8，删除实现文件 `clearance_reward.py`。
它们由表中的速度跟踪、no_progress、脚接触和正则项替代，避免多组奖励重复竞争。

表中权重为 0 的 legacy stand-still 和 standing-pose-gated action-rate 不创建。
不引入 Stones 列的周期步态、竖直速度、石柱、软着陆、终止惩罚。
**upper-body-freeze (-0.004) 按用户要求移除**，腰部三轴完全交由策略控制；
走路时不惩罚腰部相对默认角度。手臂可由有限 PD 保持站姿，但腰部不属于手臂集合。

躯干/骨盆方向只是软成本，并非锁腰；PD 目标和关节限位仍允许前倾、侧倾与扭腰。
不再奖励固定身体高度，以免压制上下楼时的重心调整。
