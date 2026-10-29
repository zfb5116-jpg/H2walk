# MuJoCo H2 20 cm stairs branch

This is the MuJoCo-only training entry point for the H2 project. The policy
controls the 12 leg joints from `CtrlLab/.../assets/robot.py`:

```text
left/right hip pitch, roll, yaw, knee, ankle roll, ankle pitch
```

The waist and both arms are locked at the H2 standing pose. They are not in
the action space and their joint positions/velocities are restored after every
MuJoCo physics step. The terrain feature is an analytical, privileged height
profile sampled in front of the pelvis, so this first branch intentionally
trains with ground-truth terrain input.

The checked-in XML is a lightweight lower-body H2-compatible model so the
branch can be smoke-tested without downloading another repository. For a
hardware-faithful run, set `CTRLLAB_H2_MJCF` or pass `--model-path` to the
official Unitree H2 MJCF. That model must expose the same 12 lower-body joint
names; upper-body joints are optional and are locked when present. The checked-
in staircase geometry is used when no external model is supplied.

## Install and run

```powershell
python -m pip install -r mujoco_stairs/requirements.txt
python mujoco_stairs/train.py --terrain-mode curriculum --total-timesteps 10000000
python mujoco_stairs/evaluate.py logs/mujoco_h2_stairs/h2_stairs_lowerbody_ppo.zip --episodes 20
```

To train directly on the final target from the first episode:

```powershell
python mujoco_stairs/train.py --terrain-mode 20cm
```

The curriculum reaches exactly 0.20 m and the evaluation script always uses
the fixed 20 cm staircase. Success means the pelvis passes the sixth 35 cm
tread without falling.
