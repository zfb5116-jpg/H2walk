# H2walk

Unitree H2 在 Isaac Sim 中的行走项目。

## 当前能力与目标

- 支持平地向前直行，指令速度为 0.3 m/s。
- 使用完整参考控制器与冻结学习残差共同运行，策略输出为 12 个下肢动作；上半身保持默认目标。
- 平地验收统计全部 1024 个环境的首次回合：持续 20 秒、前进 4.8–7.2 米、双脚各至少两次有效交替落脚，至少 922 个环境同时满足才通过。
- 当前源码的平地验收结果为 **952/1024（92.97%）**，达到上述标准。
- 转向、后退、侧移和楼梯尚不在已验收能力范围。楼梯开发与实验当前暂停。

最新平地验收数据、检查点摘要和视频保存在 `artifacts/flat_forward/`。检查点依赖随项目保留的冻结 LiDAR 重建权重与机器人资产。

## 运行环境

使用 Isaac Sim 自带 Python，并准备兼容的 Isaac Lab 与 RSL-RL 环境。本次环境使用 PyTorch 2.7.0+cu128、RSL-RL 2.3.3、Gymnasium 1.2.0；完整版本记录见验收资料。

在运行前配置本机路径：

```bash
export H2_ISAAC_SIM=/path/to/isaac-sim
export H2_ISAAC_LAB=/path/to/IsaacLab
export H2_RL_PACKAGES=/path/to/python3.11/site-packages
```

## 平地回放

在仓库根目录执行：

```bash
bash tools/replay_flat.sh
```

也可指定一个尚不存在的输出目录：

```bash
bash tools/replay_flat.sh /path/to/new-evaluation-directory
```

回放使用当前仓库内容、保存的归一化和固定检查点，不更新权重。视频显示第 0 个环境，全部 1024 个环境仍参与物理统计。验收结论只适用于上述平地前进条件。
