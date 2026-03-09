# EngineAI RL vs RSL-RL (IsaacLab) — Detailed Comparison

**Reference codebases (original, unmodified from provider):**
- `engineai_rl_workspace_ori` — PM01-specific RL workspace (`/home/armmarov/work/robot/engineai/engineai_rl_workspace_ori/`)
- `engineai_legged_gym` — ZqSA01 legged_gym implementation (`/home/armmarov/work/robot/engineai/engineai_legged_gym/`)

Both codebases share the same architecture but differ in some values. Key differences noted below.

---

## 1. PPO Algorithm

Both implementations use **identical PPO math**: surrogate loss, clipped value loss, entropy bonus, GAE, and KL-adaptive learning rate schedule.

```python
# Surrogate loss (identical in both)
ratio = exp(new_log_prob - old_log_prob)
surrogate = -advantages * ratio
surrogate_clipped = -advantages * clamp(ratio, 1-eps, 1+eps)
loss = max(surrogate, surrogate_clipped).mean()

# KL-adaptive LR (identical in both)
if kl > desired_kl * 2: lr /= 1.5
if kl < desired_kl / 2: lr *= 1.5
# clamped to [1e-5, 1e-2]
```

**legged_gym addition:** Symmetry loss (`sym_loss=True, sym_coef=1.0`) — enforces left-right mirror symmetry via observation/action permutation matrices. Not present in rl_workspace_ori.

**Verdict:** Algorithm is the same. Symmetry loss is an optional enhancement.

---

## 2. Network Architecture

| Aspect | rl_workspace_ori (PM01) | legged_gym (ZqSA01) | Our IsaacLab |
|--------|------------------------|---------------------|-------------|
| Actor dims | **[256, 256]** | **[512, 256, 128]** | [512, 256, 128] |
| Critic dims | [768, 256, 128] | [768, 256, 128] | [768, 256, 128] |
| Activation | ELU | ELU | ELU |
| Actor input | 252 (42×15 history) | 705 (47×15 history) | 64 (no history) |
| Critic input | ~345 (115×3 history) | 219 (73×3 history) | 64 (same as actor) |
| Normalizer | Baked into network `forward()` | Not used | Applied externally |
| Init | Optional orthogonal (gain=sqrt(2) hidden, 0.01 output) | Default | PyTorch defaults |
| Min std clamp | Yes (per-joint, based on joint limits) | No | No |
| init_noise_std | 1.0 | 1.0 | 1.0 |
| Symmetry loss | No | **Yes** (coef=1.0) | No |

**Key differences:**
- rl_workspace_ori actor is smaller [256,256], legged_gym uses [512,256,128]. Both much larger effective input (252/705 vs our 64) due to history.
- EngineAI can clamp minimum std per-joint based on joint range, preventing exploration from collapsing too early.
- legged_gym has symmetry loss enforcing left-right mirror consistency.

---

## 3. Observation Space — MAJOR GAP

### rl_workspace_ori PM01 Actor (obs_history_length = 15)

```
Per-step obs: dof_pos(12) + dof_vel(12) + actions(12) + base_ang_vel(3) + base_euler_xyz(3) = 42
History: 42 × 15 = 630
Goals (appended, no history): gait_phase(2) + commands(3) = 5
Total actor input: ~635 dimensions
```

### legged_gym ZqSA01 Actor (frame_stack = 15)

```
Per-step obs: command_input(5: sin/cos phase + 3 vel cmds) + dof_pos(12) + dof_vel(12) +
              last_actions(12) + base_ang_vel(3) + base_euler_xyz(3) = 47
History: 47 × 15 = 705
Total actor input: 705 dimensions
```

The actor does **NOT** see: `base_lin_vel`, `projected_gravity`, `ref_joint_diff`, `contact_mask`.
It infers velocity/contact from 15-step temporal patterns — critical for sim-to-real transfer.

### rl_workspace_ori PM01 Critic (obs_history_length = 3, privileged)

```
Per-step obs: base_lin_vel(3) + dof_pos(12) + dof_vel(12) + actions(12) +
              dof_pos_ref_diff(12) + base_ang_vel(3) + base_euler_xyz(3) +
              rand_push_force(2) + rand_push_torque(3) + terrain_frictions(2) +
              body_mass(1) + stance_curve(2) + swing_curve(2) + contact_mask(2)
              + height_measurements(17, rough terrain only)
History: ×3
Total critic input: ~345 (rough) or ~294 (flat)
```

### legged_gym ZqSA01 Critic (c_frame_stack = 3, privileged)

```
Per-step obs: command_input(5) + dof_pos(10) + dof_vel(10) + actions(10) +
              ref_dof_pos_diff(10) + base_lin_vel(3) + base_ang_vel(3) +
              base_euler_xyz(3) + rand_push_force(3) + rand_push_torque(3) +
              env_friction(1) + body_mass(1) + stance_mask(2) + contact_mask(2)
History: ×3
Total critic input: 219 (73 × 3)
```

The critic gets **privileged information**: push forces, friction, mass, ground-truth (unlagged) obs.

### Our IsaacLab (single frame, shared obs)

```
Actor AND Critic get the same 64-dim observation:
lin_vel_b(3) + ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) +
joint_vel(12) + prev_actions(12) + commands(3) + sin_phase(1) + cos_phase(1) +
ref_diff(12) + contact_mask(2) = 64
```

### Comparison

| Aspect | rl_workspace_ori | legged_gym | Our IsaacLab |
|--------|-----------------|------------|-------------|
| Actor obs history | **15 timesteps** | **15 timesteps** | **1 (none)** |
| Critic obs history | **3 timesteps** | **3 timesteps** | **1 (none)** |
| Actor total input dim | **~635** | **705** | **64** |
| Actor sees lin_vel? | **No** (inferred) | **No** (inferred) | **Yes** (direct) |
| Actor sees projected_gravity? | **No** | **No** | **Yes** |
| Actor sees base_euler_xyz? | **Yes** | **Yes** | **No** (uses proj gravity) |
| Critic privileged info? | **Yes** | **Yes** | **No** (same as actor) |
| Obs noise | **Yes** (level=1.5) | **Yes** (level=1.5) | **No** |
| Obs lag simulation | **Yes** (motor/IMU) | **Yes** (motor/IMU) | **No** |
| Manual obs_scales | **Yes** | **Yes** | **No** |

### EngineAI Observation Noise Scales

```python
noise_level = 1.5
scales = {
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
    "dof_pos": 0.02,
    "dof_vel": 2.5,
    "base_euler_xyz": 0.1,
}
```

### EngineAI Manual Obs Scales (applied before normalizer)

```python
obs_scales = {
    "base_lin_vel": 2.0,
    "base_ang_vel": 1.0,
    "body_mass": 0.1,    # legged_gym divides by 10
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "base_euler_xyz": 1.0,
    "height_measurements": 5.0,
}
```

---

## 4. Reward Functions

### Velocity Tracking (same formula)

```python
# Both use:
rew_lin_vel = scale * exp(-sum((cmd[:2] - vel[:2])^2) / sigma)
rew_ang_vel = scale * exp((cmd[2] - ang_vel[2])^2 / sigma)
```

| Param | EngineAI (both) | Ours |
|-------|----------------|------|
| lin_vel scale | 1.4 | 1.4 |
| ang_vel scale | 1.1 | 1.1 |
| tracking_sigma | 5 | 5.0 |

### Reference Joint Position — FORMULA DIFFERS

**EngineAI (both codebases):**
```python
diff = dof_pos - ref_dof_pos
return exp(-2 * norm(diff)) - 0.2 * norm(diff).clamp(0, 0.5)
# Two-part: exponential tracking + linear penalty for large errors
```

**Ours:**
```python
ref_diff_sq = torch.mean((ref - joint_pos_rel)^2, dim=-1)   # MEAN over 12 joints
return 2.2 * exp(-2.0 * ref_diff_sq)
# Single exponential, no linear penalty term
```

**Difference:** EngineAI uses `norm` (L2) not `mean of squares`, and includes a `-0.2 * clamp(norm, 0, 0.5)` linear penalty that punishes large deviations more aggressively. We use mean-of-squares exponential only.

### Feet Air Time

**EngineAI (both):**
```python
rew = sum((air_time.clamp(0, 0.5) - 0.5) * first_contact)
rew *= (norm(cmd[:2]) > 0.1)   # ZERO reward when standing still
```

**Ours:**
```python
rew = 1.5 * sum((air_time_on_contact - 0.5) * first_contact)
# No zero-command gating, no air_time clamp
```

**Differences:** EngineAI clamps air_time to 0.5 max and gates by command magnitude.

### Feet Contact Pattern

**EngineAI (both):**
```python
reward = where(contact == stance_mask, +1.0, -0.3)   # PENALIZES mismatches
return mean(reward)
```

**Ours:**
```python
contact_match = 1.0 - abs(desired - actual)   # range [0, 1], no penalty
return 1.4 * mean(contact_match)
```

**Difference:** EngineAI gives -0.3 for wrong contact. Ours gives 0.0 (no punishment).

### Action Smoothness (EngineAI uses 2nd-order term)

**EngineAI (both):**
```python
term_1 = sum((last_actions - actions)^2)                           # 1st derivative
term_2 = sum((actions + last_last_actions - 2*last_actions)^2)     # 2nd derivative (JERK)
term_3 = 0.05 * sum(abs(actions))                                  # L1 regularization
return term_1 + term_2 + term_3
```

**Ours (updated in Run 7):**
```python
term_1 = sum((actions - prev_actions)^2)
term_2 = sum((actions + prev_prev_actions - 2*prev_actions)^2)
term_3 = 0.05 * sum(abs(actions))
return -0.003 * (term_1 + term_2 + term_3)
```

**Status:** Now matches EngineAI formula (updated in Run 7).

### Default Joint Position — FORMULA DIFFERS

**EngineAI (both):**
```python
# Separate handling for yaw/roll joints
yaw_roll = sum(abs(dof_pos[:, [0,1,6,7]] - default[:, [0,1,6,7]]))  # hip yaw/roll
joint_diff = norm(dof_pos - default_dof_pos)
return exp(-yaw_roll * 100) - 0.01 * joint_diff
# Sharp exponential penalty for yaw/roll + mild linear penalty for all joints
```

**Ours:**
```python
hip_diff = mean((joint_pos[:, [0,1,6,7]] - default[:, [0,1,6,7]])^2)
return 0.8 * exp(-hip_diff * 20.0)
# Only penalizes hip pitch/roll, less sharp (×20 vs ×100)
```

**Difference:** EngineAI uses absolute value × 100 (sharper) and includes a global joint deviation linear penalty.

### Velocity Mismatch

**EngineAI (both):** `(exp(-square(base_lin_vel_z) * 10) + exp(-norm(base_ang_vel_xy) * 5)) / 2`
**Ours:** `(exp(-sum(base_lin_vel_z^2) * 10) + exp(-sum(ang_vel_xy^2) * 5)) / 2`

**Difference:** EngineAI uses L2 norm for ang_vel_xy, we use sum of squares.

### Orientation — FORMULA DIFFERS

**EngineAI (both):**
```python
quat_mismatch = exp(-sum(abs(euler_xyz[:, :2])) * 10)
orientation = exp(-norm(projected_gravity[:, :2]) * 20)
return (quat_mismatch + orientation) / 2.0
```

**Ours:**
```python
roll_pitch_error = sum(projected_gravity[:, :2]^2)
return 1.0 * exp(-roll_pitch_error * 10.0)
```

**Difference:** EngineAI combines euler angle mismatch (using `base_euler_xyz`) AND projected gravity with scaling factor 20. We only use projected gravity with scaling 10.

### Feet Clearance — FORMULA DIFFERS

**EngineAI (both):**
```python
# Height tracking during swing phase only
target_height = swing_curve * target_feet_height  # 0.1m during swing, 0 during stance
height_error = norm((target_height - feet_heights) * swing_mask)
return height_error  # scale=1.6 (POSITIVE = rewarding low error... or penalizing?)
```

**Ours:**
```python
# Penalizes swing foot below target
clearance_error = target_feet_height - foot_heights  # positive when foot too low
clearance_penalty = clearance_error.clamp(min=0.0)
return -1.6 * mean(clearance_penalty * swing_mask)
```

**Difference:** EngineAI scales target by swing_curve (sinusoidal), creating a smooth height trajectory. We use a flat target height.

### Reward dt Scaling

**EngineAI (both):** All reward scales are multiplied by `dt` (policy dt) during init:
```python
self.reward_scales[key] *= self.dt   # time-normalization
```

**Ours:** Reward scales used as-is. **Not time-normalized.**

### Track Vel Hard (same in both EngineAI codebases)

```python
lin_vel_error = norm(vel_commands[:, :2] - base_lin_vel[:, :2])
lin_vel_error_exp = exp(-lin_vel_error * 10)   # Much sharper than tracking_sigma=5

ang_vel_error = abs(vel_commands[:, 2] - base_ang_vel[:, 2])
ang_vel_error_exp = exp(-ang_vel_error * 10)

linear_error = 0.2 * (lin_vel_error + ang_vel_error)
return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error
```

**Our implementation matches this.**

### Low Speed (same in both EngineAI codebases)

```python
# -1.0 if speed < 50% of command (too slow)
# +2.0 if speed within 50-120% of command (good)
# 0.0 if speed > 120% (too fast)
# -2.0 if moving in wrong direction (sign mismatch, highest priority)
# Only active when |command| > 0.1
```

**Our implementation matches this.**

### Complete Reward Scales — Side by Side

| Reward | rl_workspace_ori | legged_gym | Our IsaacLab | Match? |
|--------|-----------------|------------|-------------|--------|
| tracking_lin_vel | 1.4 | 1.4 | 1.4 | Yes |
| tracking_ang_vel | 1.1 | 1.1 | 1.1 | Yes |
| feet_air_time | 1.5 | 1.5 | 1.5 | Yes |
| orientation | 1.0 | 1.0 | 1.0 | Yes |
| base_height | 0.2 | 0.2 | 0.2 | Yes |
| dof_ref_pos_diff | 2.2 | 2.2 | 2.2 | Yes |
| feet_contact_number | 1.4 | 1.4 | 1.4 | Yes |
| feet_clearance | 1.6 | 1.6 | -1.6 | **Formula differs** |
| feet_distance | 0.2 | 0.2 | 0.2 | Yes |
| knee_distance | 0.2 | 0.2 | — | **Missing** |
| foot_slip | -0.1 | -0.1 | -0.1 | Yes |
| base_acc | 0.2 | 0.2 | — | **Missing** |
| vel_mismatch_exp | 0.5 | 0.5 | 0.5 | Yes |
| track_vel_hard | 0.5 | 0.5 | 0.5 | Yes |
| default_joint_pos | 0.8 | 0.8 | 0.8 | **Formula differs** |
| low_speed | 0.2 | 0.2 | 0.2 | Yes |
| action_smoothness | -0.003 | -0.003 | -0.003 | Yes |
| dof_vel | -1e-5 | -1e-5 | -1e-5 | Yes |
| dof_acc | -5e-9 | -5e-9 | -5e-9 | Yes |
| torques | -1e-10 | -1e-10 | — | **Missing** (tiny) |
| collision | — | -1.0 | — | **Missing** |
| feet_contact_forces | -0.02 | -0.02 | — | **Missing** |
| lat_vel | — | — | 0.3 | **Ours only** |
| alive | — | — | 0.05 | **Ours only** |
| termination | 0.0 | — | -1.0 | **Ours only** |
| energy | — | — | -0.0001 | **Ours only** |

**Note:** EngineAI uses `termination=0.0` and `alive=0.0` (disabled). They rely on reward dt scaling and proper gait enforcement instead.

---

## 5. Gait Phase / Reference Positions — CRITICAL GAP

### EngineAI (identical in both codebases)

```python
phase = episode_length_buf * dt / cycle_time    # cycle_time = 0.8s
sin_pos = sin(2*pi * phase)

# LEFT leg swing (sin_pos < 0): hip_yaw, knee, ankle
sin_pos_l = sin_pos.clone(); sin_pos_l[sin_pos_l > 0] = 0
ref[:, 2] = sin_pos_l * 0.26           # hip_yaw_l: amplitude 0.26 rad
ref[:, 3] = -sin_pos_l * 0.52          # knee_pitch_l: amplitude 0.52 rad (2x hip)
ref[:, 4] = sin_pos_l * 0.26           # ankle_pitch_l: amplitude 0.26 rad

# RIGHT leg swing (sin_pos > 0): hip_yaw, knee, ankle
sin_pos_r = sin_pos.clone(); sin_pos_r[sin_pos_r < 0] = 0
ref[:, 8] = -sin_pos_r * 0.26          # hip_yaw_r
ref[:, 9] = sin_pos_r * 0.52           # knee_pitch_r
ref[:, 10] = -sin_pos_r * 0.26         # ankle_pitch_r

ref[abs(sin_pos) < 0.05] = 0.0         # deadband near zero crossing
ref_action = 2 * ref                    # scale by 2 for action space
ref += default_dof_pos                  # add default standing pose
phase[still_commands] = 0.0            # freeze phase when standing
```

**EngineAI drives 3 joints per leg (6 total):** hip_yaw + knee_pitch + ankle_pitch with **coupled amplitudes** (0.26/0.52/0.26). The knee bends at 2× the hip amplitude, creating a natural stepping motion.

### Our IsaacLab

```python
phase = (episode_time / 0.8) % 1.0
sin_phase = sin(2*pi * phase)

# Only drives hip_pitch (1 joint per leg)
sin_pos_l = sin_phase.clone(); sin_pos_l[sin_pos_l > 0] = 0
ref[:, 0] = sin_pos_l * 0.26           # hip_pitch_l ONLY

sin_pos_r = sin_phase.clone(); sin_pos_r[sin_pos_r < 0] = 0
ref[:, 6] = -sin_pos_r * 0.26          # hip_pitch_r ONLY

ref[abs(sin_phase) < 0.05] = 0.0
```

### Comparison

| Aspect | EngineAI (both) | Our IsaacLab |
|--------|----------------|-------------|
| Cycle time | **0.8s** | **0.8s** |
| Joints driven per leg | **3** (hip_yaw, knee, ankle) | **1** (hip_pitch only) |
| Hip joint driven | **hip_yaw** (idx 2/8) | **hip_pitch** (idx 0/6) |
| Knee in reference | **Yes** (0.52 rad, 2× hip) | **No** |
| Ankle in reference | **Yes** (0.26 rad) | **No** |
| Deadband | `abs(sin) < 0.05 → 0` | `abs(sin) < 0.05 → 0` |
| Phase on zero cmd | **Freezes to 0** | Continues cycling |
| Ref base | `ref += default_dof_pos` | Offset from default |
| ref_action scaling | **2× ref_dof_pos** | Not used |

**This is the #1 reason the robot doesn't walk.** Without knee and ankle in the reference trajectory, the robot has no template for a full stepping motion. The `dof_ref_pos_diff` reward (scale 2.2, highest positive reward) only drives hip_pitch oscillation, not a complete gait cycle.

### Default Joint Positions (both EngineAI codebases)

```python
default_joint_angles = {
    "j00_hip_pitch_l": -0.24,      # slightly flexed
    "j01_hip_roll_l": 0.0,
    "j02_hip_yaw_l": 0.0,
    "j03_knee_pitch_l": 0.48,      # bent knee (2× hip pitch)
    "j04_ankle_pitch_l": -0.24,    # compensates knee bend
    "j05_ankle_roll_l": 0.0,
    "j06_hip_pitch_r": -0.24,
    "j07_hip_roll_r": 0.0,
    "j08_hip_yaw_r": 0.0,
    "j09_knee_pitch_r": 0.48,
    "j10_ankle_pitch_r": -0.24,
    "j11_ankle_roll_r": 0.0,
}
```

---

## 6. Domain Randomization — MAJOR GAP

### EngineAI (both codebases, nearly identical values)

| Category | Parameter | rl_workspace_ori | legged_gym |
|----------|-----------|-----------------|------------|
| **Friction** | Ground friction | [0.2, 1.3] | [0.2, 1.3] |
| | Restitution | [0.0, 0.4] | [0.0, 0.4] |
| **Mass** | Base mass offset | [-4.0, +4.0] kg | [-4.0, +4.0] kg |
| | COM displacement | [-0.06, +0.06] m | [-0.06, +0.06] m |
| | Link mass multiplier | [0.8, 1.2] | [0.8, 1.2] |
| **Motors** | PD stiffness multiplier | [0.8, 1.2] | [0.8, 1.2] |
| | PD damping multiplier | [0.8, 1.2] | [0.8, 1.2] |
| | Torque multiplier | [0.8, 1.2] | [0.8, 1.2] |
| | Motor offset | [-0.035, +0.035] rad | [-0.035, +0.035] rad |
| | Joint friction | hip/knee [0.01, 1.15], ankle [0.5, 1.3] | [0.01, 1.15] |
| | Joint armature | all [0.27, 2.0] | [0.008, 0.06] |
| **Latency** | Action lag | [2, 5] steps | [1, 10] steps |
| | Motor obs lag | [5, 15] steps | [1, 10] steps |
| | IMU obs lag | [1, 10] steps | [1, 10] steps |
| **Push** | Push interval | 8s | 8s |
| | Push force xy | 0.4 m/s | 0.4 m/s |
| | Push torque | 0.6 rad/s | 0.6 rad/s |
| **Obs Noise** | noise_level | 1.5 | 1.5 |
| | per-sensor scales | ang_vel=0.2, gravity=0.05, dof_pos=0.02, dof_vel=2.5, euler=0.1 | same |

### Our IsaacLab

**Push disturbances only (Run 11).** Velocity impulse every 8s: ±0.4 m/s linear xy, ±0.6 rad/s angular rpy. Fixed friction (1.0/1.0), no mass/COM changes, no motor noise, no lag, no observation noise.

**Impact:** Push disturbances force the robot to actively step to maintain balance, which naturally leads to walking behavior. Remaining randomization (friction, mass, lag) still needed for sim-to-real transfer.

---

## 7. Simulation & Action Handling

| Aspect | rl_workspace_ori | legged_gym | Our IsaacLab |
|--------|-----------------|------------|-------------|
| Sim frequency | **1000 Hz** (dt=0.001) | **1000 Hz** (dt=0.001) | **200 Hz** (dt=0.005) |
| Policy frequency | **100 Hz** (dec=10) | **100 Hz** (dec=10) | **50 Hz** (dec=4) |
| Action clipping | [-100, 100] | [-100, 100] | [-1, 1] (tight) |
| Action scale | 0.5 per joint | 0.5 per joint | 0.5 (global) |
| PD controller | Explicit in code with Coulomb friction | Explicit in code | IsaacLab ImplicitActuator |
| Action lag | Randomized [2-5] steps | Randomized [1-10] steps | None |
| Motor offset | Randomized [-0.035, 0.035] rad | Randomized [-0.035, 0.035] rad | None |
| Torque limits | 85% of URDF limits | 85% of URDF limits | 100% |
| Episode length | **24s** | **24s** | **20s** |

---

## 8. Command Generation & Curriculum

| Aspect | rl_workspace_ori | legged_gym | Our IsaacLab |
|--------|-----------------|------------|-------------|
| lin_vel_x range | [-1.5, 1.5] | **[-0.5, 1.5]** (asymmetric!) | [-1.0, 1.0] |
| lin_vel_y range | [-0.5, 0.5] | [-0.5, 0.5] | [-0.3, 0.3] |
| ang_vel_yaw range | [-1.5, 1.5] | [-1.5, 1.5] | [-1.0, 1.0] |
| Resampling time | 8.0s | 8.0s | 8.0s |
| Still ratio | 0.1 (10%) | — (threshold-based) | 0.1 |
| Command curriculum | **Yes** (max 1.7) | **Yes** (max 1.7) | **No** |
| Small cmd filter | norm < 0.3 → zero | norm < 0.2 → zero | None |
| Terrain curriculum | **Yes** (20 levels) | **Yes** (20 levels) | **No** (flat only) |

**legged_gym note:** Forward velocity starts at [-0.5, 1.5], biased toward forward walking. Curriculum expands to ±1.7 when tracking > 80%.

---

## 9. Key Hyperparameters — Side by Side

| Parameter | rl_workspace_ori | legged_gym | Our IsaacLab | Match? |
|-----------|-----------------|------------|-------------|--------|
| learning_rate | 1e-5 | 1e-5 | 1e-5 | Yes |
| gamma | 0.994 | 0.994 | 0.994 | Yes |
| lambda (GAE) | 0.9 | 0.9 | 0.9 | Yes |
| clip_param | 0.2 | 0.2 | 0.2 | Yes |
| entropy_coef | 0.001 | 0.001 | 0.001 | Yes |
| num_learning_epochs | 2 | 2 | 2 | Yes |
| num_mini_batches | 4 | 4 | 4 | Yes |
| max_grad_norm | 1.0 | 1.0 | 1.0 | Yes |
| desired_kl | 0.01 | 0.01 | 0.01 | Yes |
| num_steps_per_env | 60 | 60 | 48 | Close |
| max_iterations | 30,000 | **300,000** | 10,000 | **Much fewer** |
| episode_length_s | 24 | 24 | 20 | Close |
| num_envs | 4096 | 4096 | 4096 | Yes |

---

## 10. PD Gains — Side by Side

| Joint | EngineAI Kp | EngineAI Kd | Our Kp | Our Kd | Match? |
|-------|-------------|-------------|--------|--------|--------|
| hip_pitch | 70 | 7.0 | 70 | 7.0 | Yes |
| hip_roll | 50 | 5.0 | 50 | 5.0 | Yes |
| hip_yaw | 50 | 5.0 | 50 | 5.0 | Yes |
| knee_pitch | 70 | 7.0 | 70 | 7.0 | Yes |
| ankle_pitch | 20 | 0.2 | 20 | 0.2 | Yes |
| ankle_roll | 20 | 0.2 | 20 | 0.2 | Yes |

PD gains match exactly.

---

## 11. Summary — Critical Gaps Checklist

### Must Fix (blocks walking)

- [x] **1. ref_joint_pos reward uses SUM instead of MEAN** — fixed in Run 3
- [x] **2. Gait reference hip joint** — uses hip_pitch (idx 0/6) since Run 5
- [x] **3. Cycle time** — changed 0.64s to 0.8s in Run 3
- [x] **3b. Gait deadband** — added `ref=0 when |sin_phase| < 0.05` in Run 3
- [x] **4. Gait reference drives 3 joints per leg** — hip_pitch + knee + ankle (0.26/0.52/0.26 rad) since Run 8/9
- [x] **5. Phase frozen on zero commands** — added in Run 8
- [x] **6. Domain randomization (push disturbances)** — added velocity impulse pushes every 8s (±0.4 m/s lin, ±0.6 rad/s ang) matching EngineAI, Run 11

### Should Fix (improves quality)

- [x] **7. Learning rate** — changed to 1e-5 in Run 4
- [x] **8. Feet air time: zero-command gating** — added
- [x] **9. Contact pattern: mismatch penalty** — changed to EngineAI's [-0.3, +1.0] in Run 8
- [ ] **10. ref_joint_pos formula** — EngineAI uses `exp(-2*norm(diff)) - 0.2*clamp(norm,0,0.5)`, we use `exp(-2*mean(diff²))` (missing linear penalty)
- [x] **11. default_joint_pos formula** — updated to EngineAI's `exp(-abs_sum * 100)` in Run 8
- [ ] **12. orientation formula** — EngineAI combines euler angles + projected gravity; we use projected gravity only
- [ ] **13. feet_clearance formula** — EngineAI scales target by swing_curve (sinusoidal); we use flat target
- [ ] **14. Command curriculum** — start narrow, expand when tracking > 80%
- [ ] **15. Small command filter** — zero out commands with norm < 0.2-0.3
- [ ] **16. Missing reward terms:**
  - [x] `feet_clearance` (scale 1.6) — swing foot height
  - [x] `default_joint_pos` (scale 0.8)
  - [x] `feet_distance` (scale 0.2)
  - [ ] `knee_distance` (scale 0.2)
  - [x] `foot_slip` (scale -0.1)
  - [ ] `base_acc` (scale 0.2)
  - [ ] `feet_contact_forces` (scale -0.02)
  - [x] `track_vel_hard` (scale 0.5)
  - [x] `low_speed` (scale 0.2)
  - [x] `dof_vel` (scale -1e-5)
  - [x] `dof_acc` (scale -5e-9)
  - [ ] `collision` (scale -1.0, legged_gym only)
  - [ ] `torques` (scale -1e-10, negligible)
- [ ] **17. Reward dt scaling** — EngineAI multiplies all scales by policy dt

### Hyperparameter Tuning (done)

- [x] entropy_coef: 0.005 → 0.001
- [x] num_learning_epochs: 5 → 2
- [x] gamma: 0.99 → 0.994
- [x] lambda (GAE): 0.95 → 0.9
- [x] num_steps_per_env: 24 → 48 (EngineAI uses 60)
- [x] max_iterations: 3000 → 10000 (EngineAI uses 30,000-300,000)
- [x] critic_hidden_dims: [512,256,128] → [768,256,128]
- [x] tracking_sigma: 0.25 → 5.0
- [x] rew_action_smoothness: -0.005 → -0.003

### Run 31 Critical Fixes

- [x] **26. Contact detection: z-height → force-based** (Run 31)
  - EngineAI: `contact_forces[:, foot_indices, 2] > 5.0N` (physics-aware)
  - Old: `foot_z < 0.16m` (only 0.012m margin above standing height 0.148m)
  - Z-height let shuffling feet stay "in contact" — broke all stepping rewards
  - Now using IsaacLab `ContactSensor` with `track_air_time=True`
- [x] **27. PPO num_learning_epochs: 5 → 2** (Run 31, match EngineAI)
  - 5 epochs caused value loss spikes (2.5K-20K) across Runs 29-30

### Future (sim-to-real transfer)

- [ ] **18. Add observation history** (15 steps for actor, 3 for critic)
- [ ] **19. Add asymmetric actor-critic** (privileged critic with forces, friction, mass)
- [ ] **20. Add domain randomization (full):**
  - [ ] Friction randomization [0.2, 1.3]
  - [ ] Base mass offset [-4, +4] kg
  - [ ] COM displacement [-0.06, +0.06] m
  - [ ] Link mass multiplier [0.8, 1.2]
  - [ ] PD stiffness/damping multiplier [0.8, 1.2]
  - [ ] Torque multiplier [0.8, 1.2]
  - [ ] Motor offset [-0.035, +0.035] rad
  - [ ] Joint friction per type
  - [ ] Joint armature per type
- [ ] **21. Add observation noise and lag:**
  - [ ] Observation noise (level=1.5, per-sensor scales)
  - [ ] Motor lag (5-15 steps)
  - [ ] IMU lag (1-10 steps)
- [x] **22. Add push disturbances** (every 8s, ±0.4 m/s lin, ±0.6 rad/s ang) — added Run 11
- [ ] **23. Increase sim frequency** — 200Hz to 1000Hz (dt=0.001, decimation=10)
- [ ] **24. Symmetry loss** (legged_gym: sym_coef=1.0, left-right mirror enforcement)
- [ ] **25. Terrain curriculum** — 20 levels, 7 terrain types
