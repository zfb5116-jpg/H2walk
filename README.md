# H2walk

H2 人形机器人的 Isaac Lab 任务、控制网络与 LiDAR 地形重建源码。

本分支保留 H2 观测与可选可靠性融合修复，并结合最新上游 `c6b0107` 完善控制奖励和台阶抬脚目标。楼梯任务默认使用 GT 高度扫描，重建需显式启用。没有运行训练、仿真或真机实验；不声称行走性能已经提升。

- [算法依据、开关与兼容性](docs/算法优化方案.md)
- [测试结果与验证边界](docs/验证记录.md)
- [本轮步态诊断、论文依据与后续方向](docs/步态诊断与优化.md)

CPU 合成数据检查使用 Python、PyTorch、NumPy、pytest：

```shell
python -m pytest tests -q -p no:cacheprovider
```

测试不启动 Isaac Lab、不读取训练权重、不执行优化器更新。完整环境和真实机器人部署仍需项目自身的 Isaac Lab/RSL-RL 配置与资产。

`logs/rsl_rl/stairs_l0b2_20261020` 保留为原 Go2 来源实验归档，本轮不使用。本分支的 H2 算法位于 `source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/`；源码接线和 CPU 检查不能替代 H2 训练与闭环验证。
