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

The checked-in XML is a lightweight surrogate, not a hardware-faithful H2.
Use the same model as the original training run when continuing a checkpoint.
An external MJCF passed with `--model-path` or `CTRLLAB_H2_MJCF` must expose
the same 12 leg joints and include physical scene geoms named `stair_01` through
`stair_06`, at the bundled staircase's positions. A robot-only MJCF is rejected:
analytical terrain observations cannot replace physical staircase collisions.
The optional `stair_landing` geom extends the final tread from x=3.0 to 4.3 m.
Upper-body joints, when present, are locked. The surrogate uses a rigid upper
mass and a three-axis waist; it no longer puts 17 joints on one body, which
MuJoCo cannot compile.

## Install and train in order

```powershell
python -m pip install -r mujoco_stairs/requirements.txt
python -m mujoco_stairs.train --terrain-mode curriculum --total-timesteps 10000000 --run-dir logs/h2_staged
```

The default curriculum is **flat -> 4/6/8/10/12/14/16/18 cm -> 20 cm**.
Each vector environment must finish at least 20 episodes at its current level;
at least 80% of its last 20 episodes must succeed before the next reset raises
the difficulty. Failures never raise difficulty. Resetting alone does not count
as success. These settings are configurable with `--curriculum-window` and
`--curriculum-success-rate`. Environments advance independently.

For explicit stages and an existing successful flat policy:

```powershell
# 1. Train flat ground from scratch (skip if a suitable flat checkpoint exists).
python -m mujoco_stairs.train --terrain-mode flat --total-timesteps 3000000 --run-dir logs/h2_flat

# 2. Load flat weights and start at low stairs; success-based progression reaches 20 cm.
python -m mujoco_stairs.train --terrain-mode curriculum --start-stage low --init-from logs/h2_flat/h2_stairs_lowerbody_ppo.zip --total-timesteps 10000000 --run-dir logs/h2_stairs

# 3. After the curriculum reaches 20 cm, specialize on fixed 20 cm stairs.
python -m mujoco_stairs.train --terrain-mode 20cm --init-from logs/h2_stairs/h2_stairs_lowerbody_ppo.zip --total-timesteps 5000000 --run-dir logs/h2_20cm
```

Step budgets are starting examples, not a guarantee of convergence. Evaluate
each stage before proceeding. Fixed low stairs are available with
`--terrain-mode low --low-step-height 0.05`. `--total-timesteps` always means
**additional** training steps; counters keep increasing after `--init-from`.
Continuation defaults to learning rate 1e-4 (new policies: 3e-4), adjustable
with `--learning-rate`. Do not overwrite the original trained checkpoint.

Checkpoints are written every `--save-every 1000000` steps, plus the final
`h2_stairs_lowerbody_ppo.zip`. Each gets a `.curriculum.json` sidecar. Loading
with `--init-from` starts the requested terrain/stage and restores the policy
and optimizer. To resume the saved curriculum level and success history, also
pass `--resume-curriculum`, using the same environment count and curriculum
settings. Episodes themselves restart.

## Swing height and foot placement

The previous maximum-of-two-foot-origin reward has been replaced by separate
sole-clearance rewards for each swinging foot:

- Near a riser, target sole height is the **next tread's world height + 5 cm**.
  It increases at every tread. A 20 cm rise targets a 25 cm lift relative to
  the previous support, rather than a fixed world or pelvis-relative height.
- Collision-geometry bounds transformed into world space measure the bottom
  of the foot, including pitch/roll, instead of the ankle body's origin.
- Only forward-moving airborne feet with the other foot supporting the robot
  earn swing reward. Static raised legs, backward swing, two-foot jumps and
  planted feet earn none. Overshooting the target reduces the reward.
- A low moving foot near a riser is penalized. Ordered, alternating tread
  contact shaping below replaces the previous one-time `landing` bonus; both
  are not applied together. Fully landed levels remain success diagnostics.
- Stair swing reward weight defaults to 2.5 (flat: 0.5), replacing the old 0.25
  height term. Tune with `--swing-reward-weight` and `--clearance-margin`.
- Positive knee/negative hip action ranges expand gradually with stair height.
  At 20 cm, knee targets can reach 1.60 rad and hip pitch -1.05 rad, within
  MJCF joint limits. The old knee maximum was 0.95 rad. Flat action mapping
  stays `home + 0.45 * action`. Observations remain 56 floats and actions 12,
  so matching old SB3 checkpoints can load; stair behavior needs fine-tuning.
- Forward reward saturates at commanded speed. Foot/riser collisions and
  prolonged stalls are penalized. The original upright/velocity/action and
  fall/success terms remain.

The analytical height profile now matches the six physical treads and landing,
including the drop after the landing. Success requires pelvis x > 3.0 m,
both feet having landed on the sixth tread, and at least one supporting foot
while upright for 0.15 s; crossing x while falling or bypassing the stairs is
not success. Flat success uses the same x/upright/support criterion.

`info` exposes target/actual sole heights, support states, highest landed stair,
and all reward components. TensorBoard logs `reward/swing_clearance`,
`reward/low_clearance`, `reward/stall`, the three stair terms, and terrain heights.

## Three stair-specific rewards adapted from WM-LOCO

Source: *World-Model-Augmented Visual Locomotion for Humanoids on
Foothold-Constrained Terrain*, arXiv:2609.02542v1, Section 2.2, Eqs. (2)--(5),
and Appendix B, Table 3: https://arxiv.org/html/2609.02542v1#S2.SS2

The paper adds three terms to a standard walking reward, shared by its PPO
baseline and world-model policy. This H2 adaptation implements the reward
ideas only; it does not reproduce the paper's G1 robot, perception, MoE
policy, training setup, world model, or reported performance.

```text
r = r_current_without_old_landing
    - 0.05 * N_boundary
    - 0.50 * N_riser
    + 0.50 * r_sequential_tread
```

The three default weights match the paper's Table 3 as initial settings, but
their numerical scale is not equivalent to the paper's simulator: raw point
counts depend on sampling density and this environment sums rewards each
control step. They need evaluation/tuning on the intended H2 physics model.

1. **`stair_boundary`** counts foot volume samples in forbidden side strips
   near the stairs. The bundled physical stairs are 4 m wide. To discourage
   detouring within that width, the H2 version constrains a central walking
   lane, default y in [-0.60, +0.60] m, from x=0.70 m to the end of the landing.
   This lane is an adaptation, not the paper's exact per-tile boundary layout.
2. **`stair_riser`** counts foot volume samples inside 4 cm-thick cuboids
   centered on each vertical riser. The upper 4 cm of each riser is excluded
   as in the paper, avoiding false penalties on near-edge tread contacts.
   A 4 cm curriculum riser therefore has no active riser cuboid; the existing
   swing-clearance reward still teaches that shallow stage.
3. **`stair_tread`** consumes tread regions in ascending order. At least 50%
   of a sole's sample points must be inside the current tread, within 2 cm
   of its surface, and that foot must have actual upward terrain contact.
   Contact starts a window `tau=clip(depth/(2*command_speed*dt), 5, 50)` steps.
   The first tread accepts either foot. Subsequent treads prefer alternation;
   a wrong-foot contact gets a 0.3 multiplier. Inside the window the raw score
   is `alternation*rho_contact - 0.7*max(rho_other-0.05, 0)`. Holding contact
   for `ceil(0.3*tau)` consecutive steps consumes the tread. Later treads earn
   nothing until the current one is consumed; consumed treads never pay again.

Explicit H2 implementation choices: each collision geom uses a 5x3x3 volume
grid and 7x5 sole grid, transformed into world space (45 and 35 points per
foot in the bundled box-foot model). Bounds approximate detailed meshes;
they are not a replacement for true mesh-surface sensors on other models.
The long top landing uses nominal 35 cm tread depth for window timing. A
window never reopens on the same tread after expiry, but later sustained
contact can consume that tread without reward to avoid deadlocking learning.
Reward sequence history resets per episode. It is not part of policy input,
so adding these terms leaves the 56-observation/12-action interface unchanged.

All three terms are exactly zero in `flat` mode and at the flat curriculum
level. They also do not pay tread reward while approaching the stairs on
flat ground. The existing walking/clearance rewards, PD controller, physical
model, and flat -> low stairs -> 20 cm curriculum stay unchanged by this patch.

Training and evaluation accept `--stair-boundary-weight`,
`--stair-riser-weight`, `--stair-tread-weight`, and `--stair-lane-half-width`.
Set all three weights to zero for a reward ablation (the superseded `landing`
bonus is not restored). TensorBoard logs `reward/stair_boundary`,
`reward/stair_riser`, and `reward/stair_tread`; `info` additionally reports
violation point counts, per-foot/tread contact ratios, completed sequential
treads, and the remaining activation window.

## Evaluate and verify

```powershell
python -m mujoco_stairs.evaluate logs/h2_flat/h2_stairs_lowerbody_ppo.zip --terrain-mode flat --episodes 20
python -m mujoco_stairs.evaluate logs/h2_stairs/h2_stairs_lowerbody_ppo.zip --terrain-mode low --low-step-height 0.08 --episodes 20
python -m mujoco_stairs.evaluate logs/h2_20cm/h2_stairs_lowerbody_ppo.zip --terrain-mode 20cm --episodes 20 --output artifacts/h2_20cm_eval.json
python -m pip install pytest
python -m pytest mujoco_stairs/tests -q
```

Evaluation is headless and reports success rate, maximum x, highest fully
landed stair, and optional per-episode reward totals. Regression tests and a
short PPO run verify code behavior; they do not establish trained 20 cm stair
success. A full training/evaluation run on the intended H2 model is required.
