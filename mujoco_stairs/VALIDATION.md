# 本地验证（2026-11-01）

Windows / Python 3.12.14 / MuJoCo 3.3.7 / PyTorch 2.15.0+cpu / SB3 2.9.2。

- 32 项自动测试：论文三项楼梯公式、交替/顺序/窗口、上下楼几何射线与解析地形高度、
  86/15 接口、腰部动态与上身重心位移、奖励时间积分、旧 checkpoint 拒绝、课程重置等。
- PPO 从头运行 1,024 transitions，保存 final/checkpoint/sidecar。
- 加载该 15 维 checkpoint，在 mixed 浅台阶追加 512 transitions，累积计数为 1,536。
- 通过 evaluate 入口分别运行浅台阶 up/down 各 2 回合，日志和 JSON 有效，均 0/2 成功。

这只是训练、续训和评估入口检查，不是上楼梯能力验证，也未进行长时间收敛训练。
旧的两份 12 维策略保持原样，不作为新版训练的起点。

命令：

```bash
python -m pytest mujoco_stairs/tests -q
python -m mujoco_stairs.train --terrain-mode flat --stair-direction mixed --total-timesteps 1024 --n-envs 2 --n-steps 256 --batch-size 128 --save-every 512 --device cpu --run-dir logs/wm_loco_waist_smoke
python -m mujoco_stairs.train --terrain-mode low --low-step-height .05 --stair-direction mixed --init-from logs/wm_loco_waist_smoke/h2_stairs_waist_ppo.zip --total-timesteps 512 --n-envs 2 --n-steps 128 --batch-size 128 --save-every 0 --device cpu --run-dir logs/wm_loco_waist_smoke_low
python -m mujoco_stairs.evaluate logs/wm_loco_waist_smoke_low/h2_stairs_waist_ppo.zip --terrain-mode low --stair-direction up --episodes 2 --output logs/wm_loco_waist_smoke_low/eval_up.json
python -m mujoco_stairs.evaluate logs/wm_loco_waist_smoke_low/h2_stairs_waist_ppo.zip --terrain-mode low --stair-direction down --episodes 2 --output logs/wm_loco_waist_smoke_low/eval_down.json
```
