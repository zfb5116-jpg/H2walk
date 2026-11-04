# H2walk 奖励函数设计调研报告

整理日期：2026-11-04。H2walk 对照基线：`0fc7722abf6f464041f26697bcb1dc3dfba069bf`，重点检查 Walk 的 `alternating_v1` 模式。本报告汇总本会话已经核对的官方代码、作者仓库及论文版本；固定提交之后的变更不在结论范围内。

## 1. 主要结论与证据边界

最直接的 H2 参考是 Unitree 官方 `unitree_rl_mjlab`：可以核对到 H2 的完整任务配置与实际启用的 16 项奖励。它适合作为 H2 基础运动约束的参照，但默认 Rough 楼梯是 0–10 cm，不能据此声称 H2 已验证 20 cm 上下楼。补足地形交互设计的资料主要来自其他机器人：RPL 提供 G1 的足底地形覆盖实现；FastStair 给出 Oli 的规划落脚轨迹；Gait-Adaptive 将步频加入 Oli 的动作空间。跨平台可以迁移设计原则，权重与实验成绩需要重新验证。

| 来源 | 平台及可核对内容 | 对 H2walk 的价值 | 本报告不作出的推断 |
| --- | --- | --- | --- |
| Unitree `unitree_rl_mjlab` | H2；完整任务、奖励代码及依赖版本 | 精简运动目标、接触相位、姿态与落地约束 | 不能把默认 Rough 配置当成 H2 高楼梯成功证据 |
| NVIDIA AGILE 教程 | 文档给出 H2 MDP；核对的公开仓库缺少对应 H2 任务 | 观测历史、奖励分类和调参顺序 | 文档列项不能还原完整 H2 数值配方 |
| RPL | G1；论文及训练代码 | 足底采样与地形高度间隙惩罚 | G1 权重不直接适用于 H2 |
| FastStair | Oli 31 自由度；论文及项目网页 | 起落脚轨迹、地形规划与策略分工 | 核对时项目仓库不含完整训练控制代码 |
| Gait-Adaptive | Oli 31 自由度；论文及项目网页 | 地形感知与步态时间的联合建模 | 增加步频需要改动作、观测和检查点协议 |

在已检查的来源中，没有找到同时满足“Unitree H2、完整开源楼梯训练实现、明确上下楼验证”的论文与仓库组合。这是本次检索结果，不代表此类工作绝对不存在。H1、H1_2、G1 不能按名称近似当成 H2；论文标题中的 “H2” 也需要核对机器人型号。

## 2. 官方 Unitree H2：16 项奖励如何组合

固定版本：[`unitree_rl_mjlab@1425b15f`](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790)，依赖 `mjlab==1.2.0`。下面的权重来自任务的实际连接关系，未把仅定义但未启用的函数算入。来源：[H2 配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/config/h2/env_cfgs.py)、[共用任务配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/velocity_env_cfg.py)、[奖励实现](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/mdp/rewards.py)。

| 配置项 | 权重 | 设计意图与计算对象 |
| --- | ---: | --- |
| `track_linear_velocity` | +1 | 指令平面速度误差，同时抑制垂直速度 |
| `track_angular_velocity` | +1 | 偏航速度跟踪，同时抑制横滚、俯仰角速度 |
| `body_orientation_l2` | −1 | torso 投影重力的水平分量平方 |
| `pose` / `variable_posture` | +1 | 按运动速度调节容许关节姿态偏离 |
| `body_ang_vel` | −0.05 | torso 世界坐标横滚、俯仰角速度平方 |
| `angular_momentum` | −0.025 | 整体角动量平方 |
| `is_terminated` | −200 | 非超时失败 |
| `joint_acc_l2` | −2.5×10⁻⁷ | 关节加速度平方 |
| `joint_pos_limits` | −10 | 超出软关节位置范围 |
| `action_rate_l2` | −0.05 | 相邻动作差平方 |
| `foot_gait` / `feet_gait` | +0.5 | 实际接触与左右参考相位一致 |
| `foot_clearance` / `feet_clearance` | −1 | 足部世界高度偏离参考值，按水平脚速加权 |
| `foot_slip` | −0.25 | 接触脚水平速度平方 |
| `soft_landing` | −0.001 | 首次落地接触力 |
| `stand_still` | −1 | 静止指令下的关节姿态偏离 |
| `self_collisions` | −1 | 自碰撞接触历史计数 |

设平面速度指令为 \(c_{xy}\)，实际速度为 \(v\)，角速度为 \(\omega\)。两个跟踪项可以写为：

\[
r_v=\exp\left[-\frac{\|c_{xy}-v_{xy}\|^2+2v_z^2}{0.25}\right],
\quad
r_\omega=\exp\left[-\frac{(c_\omega-\omega_z)^2+0.05\|\omega_{xy}\|^2}{0.5}\right].
\]

姿态项使用 \(\exp[-\operatorname{mean}_j((q_j-q_{0j})^2/\sigma_j^2)]\)，其中关节容许范围随指令速度变化。参考步态周期 0.6 s，左右相差半周期，支撑相位比例 0.56。足部高度项为 \(\sum_f |z_f^{world}-0.10|\|v_{f,xy}^{world}\|\)，没有扣除局部地面高度，因此不能直接移植为 H2walk 的楼梯净空项。[函数来源](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/mdp/rewards.py)

### 默认地形、时间尺度和动作缩放

默认 H2 Rough 使用上游 `ROUGH_TERRAINS_CFG`，其中普通与倒置楼梯的高度范围为 0–0.10 m，踏面宽 0.30 m；另一个地形集合中的 20 cm 设置不等于 H2 默认配置。来源：[mjlab 1.2.0 对应地形配置](https://github.com/mujocolab/mjlab/blob/5af32e378dcb93c9e881ace83cc5a3f5d373fe60/src/mjlab/terrains/config.py)。

该任务控制步长为 0.02 s。默认奖励管理器会乘步长，所以 `is_terminated=-200` 对一次失败转移的贡献为 −4，而非 −200。比较其他项目时，应统一“连续奖励率”和“一次事件奖励”的口径。[上游奖励管理器](https://github.com/mujocolab/mjlab/blob/5af32e378dcb93c9e881ace83cc5a3f5d373fe60/src/mjlab/managers/reward_manager.py)

需要纠正一个容易误读的实现细节：资产文件定义了 `H2_ACTION_SCALE = 0.25 × effort / stiffness`，但核对的 H2 任务只导入该常量，没有赋给动作项的 `scale`；实际继承共用配置的统一 0.25。不能把资产公式算出的踝滚转 0.11875 当成该任务正在使用的缩放。[资产常量](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/assets/robots/unitree_h2/h2_constants.py)、[动作默认值](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/velocity_env_cfg.py)、[H2 接线](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/config/h2/env_cfgs.py)

对 H2walk 而言，官方方案最有价值的是相对精简的目标分工。固定相位是否优于当前按真实支撑和位移判定的交替事件，需要实验比较；世界绝对足高则应改成地形相对目标。

## 3. NVIDIA AGILE：H2 文档有用，完整配方仍缺失

NVIDIA 教程描述 `Velocity-H2-History-v0`，包含 14 个腿部和腰部动作、5 帧历史。奖励列项共 23 个：终止、平面/偏航跟踪、躯干高度/姿态、三类力矩、垂直/水平角速度、关节速度、两阶动作变化、三类关节限位、滑脚/脚滚转/双脚偏航、躯干加速度、脚间距和跳跃。部分配置以 `RewTerm(...)` 省略，不能由页面恢复完整数值。[NVIDIA H2 MDP 教程](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/tutorial_humanoid_workflow_configure_mdp.html)

本次检查的公开 [`WBC-AGILE@f5c388eb`](https://github.com/nvidia-isaac/WBC-AGILE/tree/f5c388ebc7c1b056de891e3edb8f47693a50d5d3) 文件树包含 G1、T1，未找到文档所指的 H2 任务路径。该版本代码和 [AGILE 论文](https://arxiv.org/html/2603.20147v1) 的 G1/T1 结果不能用作 H2 楼梯结果。其可借鉴之处是先建立速度跟踪和平衡，再调整风格；H2walk 不宜直接为“与教程一致”而改变为 14 动作。

## 4. RPL：把足底踩实程度变成可计算的几何代价

RPL 的平台是 G1，公开了训练配置与奖励实现。固定仓库版本为 [`1ca0250b`](https://github.com/amazon-far/RPL/tree/1ca0250ba3745381cfbfb2b72ebcf55e995076da)。其足底项对每只脚采样 10×5 个点，使用实际足姿将点变换到世界坐标，再比较足底点与同位置地形高度。[论文](https://arxiv.org/html/2602.03002v1)、[奖励函数源码](https://github.com/amazon-far/RPL/blob/1ca0250ba3745381cfbfb2b72ebcf55e995076da/rpl_training/terms/reward.py)

简化表示为：

\[
P_{sole}=\frac{1}{N}\sum_{f\in\{L,R\}}c_f\sum_{j=1}^{N}
\mathbf1[z_{f,j}^{sole}-h(x_{f,j},y_{f,j})>0.05].
\]

其中 \(c_f\) 是接触门控，源码使用竖直接触力大于 1 的条件，并有地形掩码；\(N\) 是每只脚的点数。两脚求和的输出可以达到 2。该项描述足底点悬空比例的几何代理，不测量摩擦、压力中心、关节可达性或真实承载面积。

楼梯预设的原始 `foothold` 权重为 −10，同时将 `penalty_feet_ori` 设为 −1、`penalty_stumble` 设为 −50、`penalty_contact_no_vel` 设为 −1、`tracking_lin_vel_x` 设为 +6；楼梯边缘项权重为 0。这些是进入运行时步长与惩罚课程之前的配置值。[配置来源](https://github.com/amazon-far/RPL/blob/1ca0250ba3745381cfbfb2b72ebcf55e995076da/rpl_training/config_values/reward.py)

H2walk 已有完整足姿和 GT 高度扫描，可以采用“足底采样—局部地形查询—有效性—接触门控”的结构。但现有扫描间距为 5 cm，现有 57 个足底点是边界顶点，不是均匀面积点。以近邻插值估计覆盖时，应明确分辨率、查询半径、未知区域和样本分布，不能把未观察区域填成高度 0，也不能把近邻中位数当作真实全足底接触证明。

## 5. FastStair：奖励完整摆动轨迹，而非持续保持最高抬脚高度

FastStair 使用 Oli 31 自由度平台，把落脚规划与策略跟踪结合。足部跟踪项为：

\[
r_{foot}=\exp(-10\|p_f-p_f^{des}\|_2).
\]

这里是欧氏范数，不是范数平方。期望足部位置沿起脚点、最高点和规划落脚点随相位变化，最高点不应作为整个摆动过程的恒定目标。论文三阶段中的落脚权重为 0.6/0.5/0.4，线速度项为 1/1.35/1.1，绊脚项为 −1.5/−1.5/−2。[FastStair 论文，方法与 Table I](https://arxiv.org/html/2601.10365v1)

核对的 [`FastStair@49c254fe`](https://github.com/npcliu/FastStair/tree/49c254fe4ee124db941a49c6488e93423079bd95) 为项目网页和媒体，未提供完整训练控制实现。因此可以研究公式及轨迹结构，不能宣称已有可直接移植的完整开源控制器。

H2walk 的 `nearest_latched` 已解决部分摆动目标保持问题，但主要保存上楼目标 Z。要统一上下楼，需要把起脚、路径净空和落脚高度分开：上楼越过台阶后收敛到高踏面；下楼保留跨过上缘的净空，再向低踏面下降；普通地形落脚高度接近起点。锁定目标应有生命周期，不能在空中反复追逐新的扫描目标。

## 6. Gait-Adaptive：步频是控制变量，不能仅新增一个奖励名

该论文同样使用 Oli。动作由关节控制和步频组成，相位按 \(\phi_t=(\phi_{t-1}+\Delta t f_t)\bmod1\) 更新，左右偏移 0.5，频率限制为 0.7–1.3 Hz。论文还包括支撑脚保持 \(\exp(-100\|p_f-p_{ctr}\|^2)\)，权重 0.5，以及楼梯足部平整项，权重 0.25。[论文方法与奖励表](https://arxiv.org/html/2512.07464v1)

核对的 [项目仓库版本](https://github.com/GA-PHL/ga-phl.github.io/tree/75697759d28c0210ae6a6eb8399ad9b0dee05d9b) 不包含完整策略训练实现，部分奖励表符号也缺少源码交叉核验。因此不按论文表格直接复制整套实现。H2walk 若采用自适应频率，需要同步新增动作、相位观测、限幅、网络输出与检查点协议；它应当作为独立结构实验。

## 7. H2walk 当前设计的具体问题

基线 Walk `alternating_v1` 共配置 39 个非零权重项：任务 7 项、躯干 7 项、关节及动作 13 项、脚部及接触 12 项。“39”是启用配置项数，不表示每项每时刻都产生奖励。默认已关闭 RND；旧的 0.45 s 抬脚和接触事件奖励在该模式中权重为 0。当前实现已具备有符号净位移、合格交替落脚事件、摆动实际运动门控和 `nearest_latched`，应保留这些已有成果。[基线源码](https://github.com/zfb5116-jpg/H2walk/tree/0fc7722abf6f464041f26697bcb1dc3dfba069bf)

| 当前问题 | 对行为的潜在影响 | 重设计方向 |
| --- | --- | --- |
| 平面速度、前向速度形状项、有符号位移同时存在 | 重复优化相似量，静止附近的平滑速度奖励可能掩盖推进不足 | 保留清晰的跟踪项，以独立净位移核对任务进展 |
| 摆动脚速度、全部脚速度、脚加速度、抬脚和净空并存 | 抬脚与向前摆动同时受到抑制 | 将摆动引导和支撑滑移按状态分工，减少重复代价 |
| 固定躯干高度代理、关节偏离和姿态项同时较强 | 对上下楼必要的身体调节形成额外阻力 | 用可解释的姿态范围与必要限位，避免多重固定模板 |
| 时间方差项与真实交替事件并存 | 时长整齐不等于有效迈步 | 交替判定优先看支撑、位移、净推进和事件时间界限 |
| 支撑质量主要是接触/近地面不一致代理 | 单点接触或踩边无法充分区分 | 引入带有效性掩码的足底地形采样代理 |
| 上楼目标主要是保持的 Z | 下楼缺少低踏面落脚和过程净空的统一表达 | 使用起脚—净空—落脚目标结构 |

这些是从目标结构推导的冲突风险，不能单凭项数断言训练变差的唯一原因。此前训练证据表现为总回报和存活时间上升，同时实际推进指标下降、成功指标为零；这支持检查目标与任务不一致，但不能证明某一个权重就是全部原因。原始训练日志中的跟踪误差还受到固定累计分母影响，不能直接将曲线升高解释为速度跟踪恶化。详见项目已有 `20261030训练诊断与步态优化.md`。

## 8. 推荐的奖励架构及研究到实现的映射

建议把奖励按职责拆成四组，所有普通地形、上楼和下楼阶段共用同一套函数与权重，通过地形和目标数据变化适应场景：

1. **任务推进**：速度/偏航跟踪与有符号实际位移。位移不按正方向截断，往返必须抵消，重置和瞬移不能计入推进。
2. **步态与摆动**：真实支撑起始、连续对侧支撑、有效离地时长、足部前移和机体推进共同判定交替落脚。摆动目标在起脚时锁定，下降段允许收脚落地，静止抬脚不获得持续正收益。
3. **地形与接触**：足底几何覆盖代理、支撑滑移、绊脚和落地冲击；未知扫描单独标记有效性，不能伪装成平地或优质支撑。
4. **运动约束**：躯干姿态、必要姿态范围、动作变化、关节限位和非足部接触。各项应负责不同风险，避免三四个项重复抑制同一摆动动作。

实现应先复用已有 PPO、29 动作、历史观测和 GT 扫描接口。更换奖励模式应显式记录参数和配置指纹；旧检查点保留为历史实验，不通过删除归一化统计或修改标记强制迁移。本轮具体模块、权重和兼容决策由配套的[《20261104算法与奖励架构重设计》](20261104算法与奖励架构重设计.md)记录，本报告不代替实现或验证记录。

## 9. 验证和验收口径

CPU 测试适合验证公式、坐标系、左右脚顺序、目标生命周期、重置隔离、无效扫描处理、往返位移抵消和事件奖励对步长的一致性。它验证实现的性质，不能证明机器人会走路。

以后获准开展动态实验时，应分开报告普通地形、上楼和下楼：冻结策略下的成功率、失败率、指令积分与实际有符号位移、双脚落脚次序、每只脚的净高度变化、滑移、碰撞、力矩/动作饱和。总奖励、存活时间和训练轮数只作辅助信息。先以同条件对照区分旧模式、精简模式和几何/轨迹模块的贡献，再判断是否需要自适应步频或改变感知网络。

本次调研与整理不启动训练、仿真或真机，不执行策略，不加载保存权重或 pickle，不调用 `optimizer.step`。上述方案的行走效果仍需后续授权后的动态实验验证。
