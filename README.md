# H2walk

Unitree H2 在 Isaac Sim 中的平地行走与上楼学习项目。

## 当前能力与目标

- 完整参考控制器与冻结学习残差系统已通过平地向前行走验收：1024 个环境的首次 20 秒回合中，952 个同时满足存活、前进 4.8–7.2 米和双脚各至少两次有效交替落脚，指令速度为 0.3 m/s。
- 直接输出十二个下肢关节目标的自主平地策略已有独立的 926/1024 验收记录，用作当前上楼训练的亲本。
- 上楼目标为六级高 15 cm、深 35 cm、宽 2.4 m 的真实台阶。1024 个环境中至少 922 个须在首次完整 90 秒中同时满足存活与接触力检查、每级有效双脚交替落脚和持续承重、净升高 0.9±0.1 m，以及顶部平台稳定至少 2 秒。当前尚未通过该验收。

上楼训练使用现有激光雷达重建地形输入、五帧身体状态和五维运动上下文，保留原物理与落脚标准。训练包括真实高度课程、接触与局部支撑反馈，以及倾斜、动作变化和整脚净空代价。有限执行记录旁录策略实际使用的地形输入，用于核对地形呈现与动作响应；输入通路存在和训练指标改善均不能替代上楼验收。

## 本次提供的两份模型记录

| 记录 | 模型 | 说明 |
| --- | --- | --- |
| 平地行走 | `artifacts/flat_forward/model_999.pt` | 原 952/1024 平地验收记录。控制系统包含参考控制器和冻结学习残差，PT 本身不是完整控制器。 |
| 最新上楼训练 | `artifacts/stair_current/model_500.pt.xz.parts/` | 无损分段归档，合并解压后为 `model_500.pt`。这是本次训练最新已保存的检查点；固定 15 cm 目标组连续三个完整窗口均无合格上阶落脚，训练已按既定规则结束，尚未通过上楼验收。 |

当前训练模型、参数、对应的字节保全回执、进度快照、审查摘要与对应的完整源码快照位于 `artifacts/stair_current/`。保存轮次、来源和 SHA256 见该目录的 `record.json`。平地记录保留其原验收数据和环境资料。视频不随本次内容提供；原始历史回执中的视频路径仅是原记录的引用。

最新训练 PT 以约 314 MB 的 XZ 无损归档提供，解压后恢复 461,642,331 字节的原始 PT。为适应上传连接，归档保存在 1199 个小片段中；这些片段合计仍是一份模型记录。已核对解压后的 SHA256 与原检查点完全一致。克隆后按路径顺序合并，再解压：

```bash
cat artifacts/stair_current/model_500.pt.xz.parts/*/part* > artifacts/stair_current/model_500.pt.xz
xz -dk artifacts/stair_current/model_500.pt.xz
sha256sum artifacts/stair_current/model_500.pt
```

原始 PT 的 SHA256 应为 `903ee671568a6aba7a6414a1b059e5d66a68b7c45a1537bdffeb308d8c3fa52f`。文件保存与无损传输检查不代表模型通过上楼验收；初始检查点的 CPU 回执单独保留在 `startup/`，不作为这份最新检查点的验证结论。

当前记录也包含实际执行旁录的无损副本，保存于 `execution_records/`。这些资料用于后续核对地形输入与动作响应；本次上传只检查字节完整性，没有重新解析统计或开展仿真实验。恢复方式：

```bash
cat artifacts/stair_current/execution_records/records.jsonl.xz.parts/*/part* > artifacts/stair_current/execution_records/records.jsonl.xz
xz -dk artifacts/stair_current/execution_records/records.jsonl.xz
```

## 环境与前置依赖

运行需要兼容的 Isaac Sim、Isaac Lab、RSL-RL、机器人资产和冻结地形重建权重。模型记录分别保留对应的环境信息与源码，回放或训练应使用匹配的记录。

为保持仓库当前版本仅提供上述两份模型记录，冻结地形重建权重和直接平地亲本权重没有重复附带。依赖的路径、哈希和来源见 `artifacts/dependencies.json`。地形重建权重已存在于原仓库历史中，可按其原有字节恢复：

```bash
mkdir -p logs/rsl_rl/stairs_l0b2_20261020
git show 0d58cd35fe65fab2f85788dfb30a0bf4011e39a7:logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt > logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt
```

完整训练的其它亲本依赖须按记录另行准备。发布资料不自动启动仿真，不续训已有检查点，也不将初始模型验证或局部训练反馈视作上楼成功。
