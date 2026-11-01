# H2 MuJoCo：WM-LOCO 楼梯奖励与主动腰部控制

当前入口为 `CtrlLab-H2-Stairs-MuJoCo-v1`，策略控制 **12 个腿关节 + 3 个腰关节**。
新版策略是 **86 维观测、15 维动作**，需要重新训练；旧的 56/12 策略不能直接续训。
Isaac Lab 的训练任务不在这次改动范围内，见 `source/CtrlLab/...` 的独立配置。

奖励依据 WM-LOCO 第 2.2 节及附录 B 表 3 的 **Stairs** 列：
https://arxiv.org/html/2609.02542v1#A2

实现与论文的逐项对应、公式、适配及删除项见 [REWARDS.md](REWARDS.md)。
这是奖励设置的 H2/MuJoCo 适配，没有实现论文的世界模型或视觉编码器，也没有复现其性能。

## 腰部如何控制

动作顺序固定为左腿 6 维、右腿 6 维、腰 yaw/roll/pitch 3 维。
腰部目标范围为 yaw ±0.5、roll ±0.35、pitch ±0.6 rad，实际受 MJCF 关节限位裁剪。
通过有限 PD 力矩驱动，不再逐帧恢复腰部 qpos/qvel，也不施加走路时的腰部姿态锚定奖励。
躯干和骨盆方向奖励都是软惩罚，允许为楼梯动作适度倾身；失稳终止规则仍在。

自带 XML 把腰轴放到 `torso` 根部，整个上身质量随腰部移动，支持调整整体重心。
自带模型是简化碰撞模型，不是硬件精确的 H2。外部模型必须提供相同 15 个控制关节。
外部模型存在手臂关节时，用有限 PD 保持手臂站姿，腰关节不包含在这个手臂集合中。

观测包括骨盆局部线速度/角速度、骨盆/躯干投影重力、15 维关节位置和速度、
15 维上一步动作、vx/vy/wz 指令、航向误差 sin/cos，以及 24 个相对高度采样。
地形输入仍为解析真值，不是相机或 LiDAR。

## 训练

```bash
python -m pip install -r mujoco_stairs/requirements.txt
# 新策略从头学平地；不要加载旧 12 维策略。
python -m mujoco_stairs.train --terrain-mode flat --stair-direction mixed --total-timesteps 3000000 --run-dir logs/h2_wm_flat
# 评估通过后，用同一 15 维接口进入浅台阶课程。
python -m mujoco_stairs.train --terrain-mode curriculum --start-stage low --stair-direction mixed --init-from logs/h2_wm_flat/h2_stairs_waist_ppo.zip --total-timesteps 10000000 --run-dir logs/h2_wm_stairs
```

以上步数是运行示例，不能保证收敛。训练初始前进速度为 0.15–0.45 m/s。
`--stair-direction up/down/mixed` 控制上楼、下楼或每回合随机上下楼。
下楼从抬高平台开始，沿 +X 走下 6 级台阶；上楼从地面开始沿 +X 登上 6 级。
`mixed` 是同一个策略在两个方向训练，不是单回合先上后下。

地形模式：

- `flat`：纯平地。
- `low --low-step-height 0.05`：固定浅台阶，不会升级。
- `20cm`：固定 20 cm 台阶，不会升级。
- `curriculum`：平地、4/6/8/10/12/14/16/18/20 cm，达到窗口成功率才升级。

课程沿用每环境最近 20 回合至少 80% 成功的规则，可用参数调整；失败和重置本身不升级。
默认 mixed 课程窗口汇总上下楼回合，并未分别要求两个方向达标，因此正式评估应分开运行 up/down。
同一台阶的顺序接触必须持续至少 `ceil(0.3*tau)` 帧才算完成。
成功要求按顺序走过 6 个接触区域、身体到达终点、位于行走通道内、脚有支撑并保持稳定 0.2 s。
成功和摔倒只是完成标志，不再发额外奖励。

每个 checkpoint 旁写 `.curriculum.json`，包含奖励版本、完整权重、控制关节顺序、
PD/动作尺度、模型 SHA256、控制周期、启动参数以及各环境课程状态。
`--init-from` 要求相同模型/控制契约；新阶段不会继承旧阶段的课程窗口。
真正恢复课程还需 `--resume-curriculum`、相同环境数量与课程设置。
`--total-timesteps` 表示本次追加步数。

## 评估与验证

```bash
python -m mujoco_stairs.evaluate logs/h2_wm_stairs/h2_stairs_waist_ppo.zip --terrain-mode 20cm --stair-direction up --episodes 20 --output artifacts/up.json
python -m mujoco_stairs.evaluate logs/h2_wm_stairs/h2_stairs_waist_ppo.zip --terrain-mode 20cm --stair-direction down --episodes 20 --output artifacts/down.json
python -m pip install pytest
python -m pytest mujoco_stairs/tests -q
```

评估报告成功率、前进距离、顺序完成的台阶数和逐项奖励。
训练日志包含全部 29 个非零权重项。奖励统一乘控制 dt=0.01 s；图表中逐步数值是积分后的奖励。
删除了旧的 `--clearance-margin`、`--swing-reward-weight` 和临时权重开关，
奖励设置统一维护在 `mujoco_stairs/rewards.py`，避免命令与配置分叉。

上下楼的外部 MJCF 场景需提供 `stair_01..06`、`stair_landing`、`stair_start`，
位置与自带模型一致；`up` 模式不强制需要 `stair_start`。
不能只提供机器人模型并用高度观测代替真实地形。
立面顶部排除 4 cm 与论文一致，因此不超过 4 cm 的浅台阶没有立面几何惩罚；
顺序踏面奖励、脚滑惩罚和其他行走奖励仍然有效。
