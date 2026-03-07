# PM01 Walking Environment

## Overview

A bipedal walking environment for the EngineAI PM01 humanoid robot built on
IsaacLab's Direct RL workflow. The robot learns to walk by tracking velocity
commands while following a sinusoidal gait reference.

**Gym ID:** `Isaac-PM01-Walking-Direct-v0`

| Parameter | Value |
|-----------|-------|
| Physics rate | 200 Hz (dt = 0.005 s) |
| Policy rate | 50 Hz (decimation = 4) |
| Parallel envs | 4096 |
| Episode length | 20 s |
| Action space | 12 (leg joints, continuous) |
| Observation space | 64 (continuous) |


## Architecture

```
__init__.py              ← gym.register() — entry point
armrobotlegging_env_cfg.py  ← ArmrobotleggingEnvCfg (all settings)
armrobotlegging_env.py      ← ArmrobotleggingEnv (all logic)
agents/
  rsl_rl_ppo_cfg.py      ← PPO hyperparameters for RSL-RL
robots/
  pm01.py                ← PM01_CFG (ArticulationCfg + UrdfFileCfg)
  pm01_assets/           ← URDF + STL meshes
```


## Robot

**EngineAI PM01** — 24-DOF humanoid (12 leg + 1 waist + 10 arm + 1 head).
Only the 12 leg joints are controlled by the policy; all others hold default
positions via PD control.

### Controlled joints (action indices)

| Index | Joint | Effort (N·m) | Stiffness | Damping |
|-------|-------|-------------|-----------|---------|
| 0 | j00_hip_pitch_l | 164 | 70 | 7.0 |
| 1 | j01_hip_roll_l | 164 | 50 | 5.0 |
| 2 | j02_hip_yaw_l | 52 | 50 | 5.0 |
| 3 | j03_knee_pitch_l | 164 | 70 | 7.0 |
| 4 | j04_ankle_pitch_l | 52 | 20 | 0.2 |
| 5 | j05_ankle_roll_l | 52 | 20 | 0.2 |
| 6 | j06_hip_pitch_r | 164 | 70 | 7.0 |
| 7 | j07_hip_roll_r | 164 | 50 | 5.0 |
| 8 | j08_hip_yaw_r | 52 | 50 | 5.0 |
| 9 | j09_knee_pitch_r | 164 | 70 | 7.0 |
| 10 | j10_ankle_pitch_r | 52 | 20 | 0.2 |
| 11 | j11_ankle_roll_r | 52 | 20 | 0.2 |


## Action Space

**Type:** Joint position targets via implicit PD control.

```
target_position = default_joint_pos + action_scale * clamp(action, -1, 1)
```

- `action_scale = 0.5` rad
- Policy outputs 12 values in [-1, 1]
- Applied every physics sub-step (4 times per policy step)


## Observation Space (64 dimensions)

| Dims | Name | Description |
|------|------|-------------|
| 3 | `lin_vel_b` | Base linear velocity in body frame [m/s] |
| 3 | `ang_vel_b` | Base angular velocity in body frame [rad/s] |
| 3 | `projected_gravity` | Gravity vector in body frame (0,0,-1 when upright) |
| 12 | `joint_pos_rel` | Leg joint positions relative to default standing pose [rad] |
| 12 | `joint_vel` | Leg joint velocities [rad/s] |
| 12 | `prev_actions` | Previous policy output [-1, 1] |
| 3 | `commands` | Velocity commands: [vx, vy, yaw_rate] |
| 1 | `sin_phase` | sin(2 * pi * gait_phase) |
| 1 | `cos_phase` | cos(2 * pi * gait_phase) |
| 12 | `ref_joint_diff` | Reference gait joint pos - current joint pos [rad] |
| 2 | `contact_mask` | Binary foot contact: [left, right] |


## Velocity Commands

The policy receives randomized velocity commands that are periodically
resampled during training:

| Command | Range | Unit |
|---------|-------|------|
| Forward velocity (vx) | [-1.0, 1.0] | m/s |
| Lateral velocity (vy) | [-0.3, 0.3] | m/s |
| Yaw rate | [-1.0, 1.0] | rad/s |

- Resampled every **8 seconds**
- **10%** chance of zero commands (standing still)
- Also resampled on episode reset


## Gait Phase System

A sinusoidal clock drives the walking reference. This tells the policy
**when** each leg should swing or stance.

```
cycle_time = 0.64 s  (≈ 1.56 Hz gait frequency)
phase = (episode_time / cycle_time) % 1.0
sin_phase = sin(2π × phase)
cos_phase = cos(2π × phase)
```

### Reference joint positions

When a leg is in swing phase, reference positions create a stepping motion:

```
                 sin_phase < 0          sin_phase > 0
                 (left swings)          (right swings)
hip_pitch        +sin × 0.26            -sin × 0.26
knee_pitch       -sin × 0.52            +sin × 0.52
ankle_pitch      +sin × 0.26            -sin × 0.26
```

Only hip_pitch, knee_pitch, and ankle_pitch get reference offsets.
Roll and yaw joints have zero reference (stay neutral).

### Expected contact pattern

| Phase | Left foot | Right foot |
|-------|-----------|------------|
| sin >= 0 | Stance (ground) | Swing (air) |
| sin < 0 | Swing (air) | Stance (ground) |


## Reward Function

All positive rewards are **clamped >= 0** before adding penalties.
This prevents negative reward spirals during early training.

### Tracking rewards (follow commands)

| Reward | Weight | Formula |
|--------|--------|---------|
| Linear velocity tracking | +1.5 | exp(-\|cmd_xy - vel_xy\|² / 0.25) |
| Angular velocity tracking | +1.0 | exp(-\|cmd_yaw - vel_yaw\|² / 0.25) |

### Gait quality rewards

| Reward | Weight | Formula |
|--------|--------|---------|
| Gait reference tracking | +2.0 | exp(-2 × \|ref_pos - joint_pos\|²) |
| Feet air time | +1.5 | Σ (air_time - 0.5) × contact (per foot) |
| Contact pattern | +1.2 | Match expected stance/swing per gait phase |
| Orientation (upright) | +1.0 | exp(-\|roll,pitch gravity error\|² × 10) |
| Base height | +0.2 | exp(-\|height - 0.8132\| × 100) |
| Velocity mismatch | +0.5 | Penalize vertical vel + roll/pitch angular vel |
| Alive bonus | +0.15 | Constant per-step survival reward |

### Penalties

| Penalty | Weight | Formula |
|---------|--------|---------|
| Action smoothness | -0.005 | \|action - prev_action\|² + 0.5 × \|action\|² |
| Energy | -0.0001 | Σ action² × \|joint_vel\| |
| Termination | -2.0 | Applied on fall (after positive clamping) |


## Termination Conditions

An episode ends (terminated = True) when:

1. **Base too low:** `root_pos_z < 0.45 m` — robot fell
2. **Forbidden body contact:** any of these bodies touch the ground
   (`body_z < 0.03 m`):
   - `link_base` (pelvis)
   - `link_knee_pitch_l`, `link_knee_pitch_r`
   - `link_torso_yaw` (upper body)
3. **Timeout:** episode exceeds 20 seconds (time_out, not terminated)


## Reset Behavior

On reset, environments are restored to:

- Default joint positions (slight knee bend standing pose)
- Zero velocities
- New velocity commands sampled
- Gait phase, air time, and action buffers zeroed


## PPO Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (RSL-RL) |
| Actor network | [512, 256, 128] ELU |
| Critic network | [512, 256, 128] ELU |
| Learning rate | 1e-3 (adaptive schedule) |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip | 0.2 |
| Entropy coef | 0.005 |
| Steps per env | 24 |
| Mini-batches | 4 |
| Epochs | 5 |
| Max iterations | 3000 |
| Observation normalization | Empirical (running stats) |
| Initial noise std | 1.0 |


## Usage

### Training

```bash
cd /home/armmarov/work/robot/isaac/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-PM01-Walking-Direct-v0 \
    --num_envs 4096
```

### Playing a trained policy

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-PM01-Walking-Direct-v0 \
    --num_envs 32 \
    --checkpoint /path/to/model.pt
```

### Listing registered environments

```bash
python scripts/environments/list_envs.py
```


## File Reference

| File | Purpose |
|------|---------|
| `tasks/direct/armrobotlegging/__init__.py` | Gym registration |
| `tasks/direct/armrobotlegging/armrobotlegging_env_cfg.py` | All configuration (physics, rewards, commands) |
| `tasks/direct/armrobotlegging/armrobotlegging_env.py` | Environment logic (obs, rewards, resets) |
| `tasks/direct/armrobotlegging/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| `robots/pm01.py` | Robot ArticulationCfg with UrdfFileCfg |
| `robots/pm01_assets/urdf/pm01.urdf` | PM01 URDF (full body) |
| `robots/pm01_assets/meshes/*.stl` | Visual/collision meshes |


## Design Decisions

### Why Direct workflow (not Manager-Based)?

The Direct workflow gives full control over observation composition, reward
computation, and gait phase logic in a single file. For a custom bipedal
walking task with a sinusoidal gait reference, this is simpler than
wiring it through the Manager-Based observation/reward manager system.

### Why position control (not effort/torque)?

Position control with PD gains is more stable for locomotion and matches
the real PM01 motor controller. The policy learns position offsets from
the default pose, and the PD controller handles low-level torque computation.

### Why gait phase reference?

Without a gait reference, bipedal robots tend to learn shuffling or hopping
gaits. The sinusoidal reference provides a strong prior for alternating
left-right stepping, dramatically speeding up learning and producing more
natural walking patterns.

### Why clamp positive rewards >= 0?

During early training when the robot frequently falls, large negative rewards
from multiple penalty terms can create unstable gradients. Clamping positive
rewards to zero before adding the termination penalty ensures a bounded
negative signal on failure while keeping the reward landscape smooth.
