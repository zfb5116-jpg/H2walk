# H2walk

H2 人形机器人的 Isaac Lab 任务、控制网络与 LiDAR 地形重建源码。

本分支目标是让同一 H2 策略兼顾普通地形和上下楼梯。Stairs任务保留GT扫描和混合地形，采用地面相对净空与宽窄带宽混合跟踪；修正低速、yaw限幅时的奖励门控，训练按固定地形列逐组达标后扩速，并提供配套目标上限的20cm渐进档案。各项保留基线开关。main新增项经接口与CPU反例检查后选择性适配。没有运行训练、仿真或真机实验，尚未证明正常行走或20cm通过能力。

- [算法依据、开关与兼容性](docs/算法优化方案.md)
- [测试结果与验证边界](docs/验证记录.md)
- [本轮步态诊断、论文依据与后续方向](docs/步态诊断与优化.md)
- [按实际前进量扩速的命令课程](docs/命令课程优化.md)
- [新结果诊断、接触时序与足底高度参考](docs/本轮结果诊断与接触优化.md)
- [同一策略兼顾普通地形与上下楼梯](docs/普通地形与楼梯统一行走.md)
- [main对照、训练预期与本轮补强](docs/main对照与训练预期复核.md)

CPU 合成数据检查使用 Python、PyTorch、NumPy、pytest：

```shell
python -m pytest tests -q -p no:cacheprovider
```

测试不启动 Isaac Lab、不读取训练权重、不执行优化器更新。完整环境和真实机器人部署仍需项目自身的 Isaac Lab/RSL-RL 配置与资产。

`logs/rsl_rl/stairs_l0b2_20261020` 保留为原 Go2 来源实验归档，本轮不使用。本分支的 H2 算法位于 `source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/`；源码接线和 CPU 检查不能替代 H2 训练与闭环验证。
