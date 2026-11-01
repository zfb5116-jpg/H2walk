# 本地验证（2026-11-01）

Windows / Python 3.12.14 / MuJoCo 3.3.7 / PyTorch 2.15.0+cpu / SB3 2.9.2。

- 32 项自动测试：论文三项楼梯公式、交替/顺序/窗口、上下楼几何射线与解析地形高度、
  86/15 接口、腰部动态与上身重心位移、奖励时间积分、旧 checkpoint 拒绝、课程重置等。
- PPO 从头运行 1,024 transitions，保存 final/checkpoint/sidecar。
- 加载该 15 维 checkpoint，在 mixed 浅台阶追加 512 transitions，累积计数为 1,536。
- 通过 evaluate 入口分别运行浅台阶 up/down 各 2 回合，日志和 JSON 有效，均 0/2 成功。

这只是训练、续训和评估入口检查，不是上楼梯能力验证，也未进行长时间收敛训练。
旧的两份 12 维策略保持原样，不作为新版训练的起点。

## 推送前复测

- 再次运行全部测试：32 passed；compileall 与 git diff --check 通过。
- 在 5 cm mixed 场景从头运行 4,096 transitions，保存 2,048/4,096 checkpoint 和最终策略。
- 加载最终策略追加训练 256 transitions，总计数正确恢复至 4,352。
- 加载 4,096 步策略，对 5 cm/20 cm × up/down 四种组合各渲染一个 seed=0 回合。
  观测、奖励均有限，视频成功生成，29 项奖励与版本化契约匹配。
- 四个回合均失稳，顺序完成台阶数为 0；4,096 步只是入口冒烟检查，不能视为收敛策略。
- 5 cm up 回合腰 yaw/roll/pitch 最大绝对角度约为 0.081/0.189/0.180 rad，
  5 cm down 约为 0.094/0.158/0.228 rad，证明策略输出确实驱动了腰部。
- 录像和逐回合结果作为本地验证产物保留，不把冒烟模型上传为可上楼梯的训练结果。

命令：

```bash
python -m pytest mujoco_stairs/tests -q
python -m mujoco_stairs.train --terrain-mode flat --stair-direction mixed --total-timesteps 1024 --n-envs 2 --n-steps 256 --batch-size 128 --save-every 512 --device cpu --run-dir logs/wm_loco_waist_smoke
python -m mujoco_stairs.train --terrain-mode low --low-step-height .05 --stair-direction mixed --init-from logs/wm_loco_waist_smoke/h2_stairs_waist_ppo.zip --total-timesteps 512 --n-envs 2 --n-steps 128 --batch-size 128 --save-every 0 --device cpu --run-dir logs/wm_loco_waist_smoke_low
python -m mujoco_stairs.evaluate logs/wm_loco_waist_smoke_low/h2_stairs_waist_ppo.zip --terrain-mode low --stair-direction up --episodes 2 --output logs/wm_loco_waist_smoke_low/eval_up.json
python -m mujoco_stairs.evaluate logs/wm_loco_waist_smoke_low/h2_stairs_waist_ppo.zip --terrain-mode low --stair-direction down --episodes 2 --output logs/wm_loco_waist_smoke_low/eval_down.json
```
