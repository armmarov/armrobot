# Codex Review — 2026-03-23 10:00

> **Question:** We just implemented Run 47. Review the two changes in armrobotlegging_env.py: (1) orientation formula at line ~757 — is the dual-signal implementation correct? (2) default_joint_pos at line ~805 — are the splay indices [1,2,7,8] correct and is the norm+deadband formula right? Any bugs or issues before we run training?
> **Session:** `019d1855-ac62-7071-a113-06254c0ee9df` (resumed — has full gap-study context)

---

OpenAI Codex v0.116.0 (research preview)
--------
workdir: /home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/armmarov/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d1855-ac62-7071-a113-06254c0ee9df
--------
user
The following files may have been updated since your briefing — review the latest versions:

=== FILE: docs/TRAINING_PLAN.md (latest) ===
# PM01 Walking — Training Plan

This document tracks planned improvements, their motivation, implementation details, and review status.
Designed for multi-model review — each plan item should be independently evaluable.

---

## Current State (Run 46 — in progress)

| Item | Value |
|------|-------|
| Obs space | 334-dim (64 current + 15×18 compact history) |
| vel_x | ~0.47–0.63 m/s |
| Episode length | ~500 steps |
| noise_std | 0.08 (converging) |
| Key weakness | Unnatural gait: forward lean, wide hip splay, L/R force imbalance (175/237 N) |

---

## Run 47 — Natural Gait Fixes

**Goal:** Make the walking look more natural — reduce forward trunk lean and hip splay.

**Status:** Planned (implement after Run 46 finishes or if Run 46 plateaus)

**Review status (Codex + Qwen):**
- Change 1 (ref_joint_pos) reframed — live code already correct; hip splay fix belongs in `default_joint_pos`
- Change 2 (orientation) — APPROVED by both reviews; keep `rew_orientation=1.0` (Qwen Q2 answered)

---

### Change 1 — Hip splay regularization via `default_joint_pos` ~~ref_joint_pos formula~~

> **REFRAMED after Codex review:** `ref_joint_pos` at `armrobotlegging_env.py:711` ALREADY uses
> `exp(-2*norm(diff)) - 0.2*clamp(norm,0,0.5)` — the EngineAI-style formula. Do NOT rewrite it.
> The EngineAI snippet originally cited is structurally a **posture regularization** term, not a tracking reward.
> Hip splay should be addressed via `default_joint_pos` tuning or a dedicated hip-yaw/roll penalty term.

**Problem:** Hip_roll and hip_yaw splay outward → wide-stance unnatural posture. Current `rew_default_pos` (`armrobotlegging_env.py:802`) already penalizes deviation from default pose but may need higher weight or tighter deadband for hip-specific joints.

**Options (pick one to test):**
1. **Increase `default_joint_pos` weight** for hip_roll and hip_yaw indices specifically (add per-joint weighting vector)
2. **Add a dedicated hip-splay penalty** as a new separate reward term with small weight (0.1–0.3), targetting indices 1,2 (hip_roll_l, hip_yaw_l) and 7,8 (hip_roll_r, hip_yaw_r)

**Our joint layout:**
```
0: j00_hip_pitch_l    1: j01_hip_roll_l    2: j02_hip_yaw_l
3: j03_knee_pitch_l   4: j04_ankle_pitch_l  5: j05_ankle_roll_l
6: j06_hip_pitch_r    7: j07_hip_roll_r    8: j08_hip_yaw_r
9: j09_knee_pitch_r  10: j10_ankle_pitch_r 11: j11_ankle_roll_r
```

**Target indices (must include both roll AND yaw per Qwen round-2 review):**
- `hip_roll_l (1), hip_yaw_l (2)` — left splay pair
- `hip_roll_r (7), hip_yaw_r (8)` — right splay pair
- Do NOT increase weight for hip_pitch (0,6), knee (3,9), ankle (4,5,10,11) — those need freedom for gait

**Review answers (Qwen):** 0.1 rad deadband appropriate for 0.26 rad gait amplitude (Q1 answered).

**Risk:** Low–medium. Tuning existing `default_joint_pos` is safer than adding a new term. Start by doubling its weight and observe.

---

### Change 2 — Orientation formula (euler + gravity combined)

**Problem:** Current formula only uses `projected_gravity[:, :2]` — a single signal with scale factor 10. EngineAI uses two complementary signals at different scales, giving much stronger upright enforcement.

**Current code:**
```python
r = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 10)
```

**EngineAI formula** (`zqsa01.py`, line 293–295):
```python
quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
orientation   = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
r = (quat_mismatch + orientation) / 2.
```

**Implementation note:** Need `base_euler_xyz` in the env. IsaacLab provides this via:
```python
from isaaclab.utils.math import euler_xyz_from_quat
euler_xyz = euler_xyz_from_quat(self.robot.data.root_quat_w)  # [N, 3] roll, pitch, yaw
```

**Expected effect:** Stronger upright penalty → less forward trunk lean during fast walking.

**Risk:** Low. Reward is still in [0, 1] range, just uses two complementary signals. The `/2.0` denominator already normalizes the combined signal — effective range and scale are unchanged.

**Review answers:**
- `rew_orientation` weight: **keep at 1.0** — do NOT scale down. The `/2.0` average handles normalization (Qwen Q2).
- `euler_xyz_from_quat` is the correct IsaacLab API (`from isaaclab.utils.math import euler_xyz_from_quat`).

---

## Run 48 — Symmetry Loss

**Goal:** Enforce L/R mirror symmetry at the PPO level — the main EngineAI feature we're missing.

**Status:** Planned after Run 47 validates formula fixes.

### What it does

During each PPO mini-batch update, an additional loss term is computed:
```python
mirror_obs    = obs @ obs_perm_matrix         # swap L/R dims: vy→-vy, joint_l↔joint_r, etc.
mirror_action = actor(mirror_obs)             # what policy outputs for the mirrored situation
expected_sym  = action @ act_perm_matrix      # what current action looks like when mirrored
sym_loss      = (action - expected_sym).pow(2).mean()
total_loss   += sym_coef * sym_loss           # sym_coef = 1.0 in EngineAI
```

This forces: "if the left and right sides are swapped, the policy should output the mirror of the original action."

### Permutation matrices for PM01

Our obs layout (64-dim current frame, indices for permutation):
```
[0-2]  lin_vel_b:       vx,  vy,  vz   → mirror: vx, -vy,  vz
[3-5]  ang_vel_b:       ωx,  ωy,  ωz   → mirror: -ωx, ωy, -ωz
[6-8]  proj_gravity:    gx,  gy,  gz   → mirror: gx, -gy,  gz
[9-20] joint_pos_rel:   L[0..5], R[0..5] → swap L↔R, negate roll/yaw
[21-32] joint_vel:      same swap as joint_pos
[33-44] prev_actions:   same swap as joint_pos
[45-47] commands:       vx_cmd, vy_cmd, yaw_cmd → vx, -vy, -yaw
[48-49] sin/cos_phase:  unchanged (gait phase is symmetric)
[50-61] ref_joint_diff: same swap as joint_pos
[62-63] contact_mask:   L, R → swap R, L
```

For history frames (indices 64–333): each 18-dim frame = `ang_vel_b(3) + proj_gravity(3) + joint_pos_rel(12)` → same mirror rules as above, repeated 15×.

**Action permutation** (12-dim):
```
L[hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] ↔
R[hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
with sign flip on: hip_roll, hip_yaw, ankle_roll (lateral/rotational joints)
```

### Implementation approach

> **Codex review:** Do NOT copy the full RSL-RL package. Use a thin override instead to avoid long-term upstream drift.

**Preferred approach (thin fork):**
1. Create `source/rsl_rl_custom/` with ONLY `algorithms/ppo.py` (single file)
2. Subclass the installed `rsl_rl` PPO class, override only the `update()` method to inject `sym_loss`
3. Pass permutation index arrays (not dense matrices) via `rsl_rl_ppo_cfg.py`
4. Register the custom runner in IsaacLab extension

**Before implementing:** Write and verify permutation index arrays in a standalone unit test. Confirm each joint's sign before modifying any PPO code.

**Risk:** Medium-high. Wrong sign on any joint reverses the symmetry constraint. Careful per-joint sign verification is mandatory.

**Review answers:**
- `sym_coef`: use **1.0 fixed from start** — EngineAI style, no annealing needed (Qwen Q3)
- History permutation: compact 18-dim frames handled correctly by frame-stack loop when `single_obs_len=18` (Qwen Q4)
- `hip_pitch` sign: positive pitch left = positive pitch right (both flex forward) — **same sign, no flip needed**. Flip needed for: hip_roll, hip_yaw, ankle_roll (lateral/rotational)

---

## Run 49 — Domain Randomization + Observation Noise

**Goal:** Prepare for sim-to-real transfer.

**Status:** Planned after symmetry loss confirmed working.

> **Codex review:** Split into staged runs — too many coupled changes for one training loop. If metrics regress, attribution is weak.
> **Qwen review:** Friction should be FROM ITERATION 1, not after convergence. EngineAI applies friction `[0.2, 1.3]` from the start.

### Staged breakdown (revised)

**Run 49a — Observation noise + action lag** (sensor realism, lowest destabilization risk):
1. **Observation noise:** IMU noise (ang_vel ±0.05 rad/s), joint encoder noise (±0.01 rad)
2. **Action lag:** 1–2 step delay on applied actions (motor latency)

**Run 49b — Contact randomization** (friction from iter 1 per EngineAI):
3. **Friction** multiplicative `[0.2, 1.3]` — applied **from iteration 1**, not after convergence (Qwen correction).
   > Note: friction rand scaffolding **already exists** in `armrobotlegging_env_cfg.py` at range `(0.7, 1.3)` but is disabled. Run 49b = replace range with `[0.2, 1.3]` and enable from start — not adding from scratch.
   > Also: push randomization and PD-gain randomization are **already live** in config. Run 49 adds on top of these.

**Run 49c — Morphology randomization**:
4. **Mass randomization** ±20% per body link
5. **COM offset** ±2cm per axis
6. **Motor strength** ±20%

**Review answers:**
- Friction: multiplicative (not additive as in Run 42), from iter 1 (Qwen)
- Order: obs noise → latency → friction → mass/COM/strength (Codex + Qwen both recommend this order)

---

## Run 50 — Command Curriculum

**Goal:** Structured velocity learning — start easy, expand range as tracking improves.

**Status:** Planned. NOT implementation-ready — see below.

> **Codex review:** Underspecified. The trigger condition is ambiguous because current code logs per-episode reward components, not a curriculum controller state. Define metric, persistence, and logging before implementing.

### Design

```
Start:   cmd_lin_vel_x ∈ (0.2, 0.4)
Expand:  if <metric> > 0.85 × max for 200 iters → increase max by 0.1
Target:  cmd_lin_vel_x ∈ (0.3, 1.5)  (EngineAI uses up to 1.7)
```

### TODO before implementing

- **Metric definition**: Use mean `rew_tracking_lin_vel` (already logged per episode) averaged over 200 iters, normalized by current max velocity target
- **Update cadence**: Check every 200 training iterations (not every episode)
- **Persistence**: Store current curriculum level in `env_cfg` or a sidecar JSON; reload on resume
- **Logging keys**: Add `curriculum/vel_x_max` to tensorboard so curriculum state is visible

---

## Open Questions — Review Status

All questions answered by Codex + Qwen reviews.

1. ~~**Run 47 ref_joint_pos deadband**~~: ANSWERED — 0.1 rad is appropriate for 0.26 rad gait amplitude. But Change 1 reframed: target `default_joint_pos`, not `ref_joint_pos`.

2. ~~**Run 47 orientation weight**~~: ANSWERED — Keep `rew_orientation=1.0`. The `/2.0` denominator normalizes; no downward adjustment needed.

3. ~~**Run 48 symmetry coef**~~: ANSWERED — Use 1.0 fixed from start (EngineAI style). No annealing.

4. ~~**Run 48 history permutation**~~: ANSWERED — Frame-stack permutation loop handles compact 18-dim frames correctly when `single_obs_len=18`.

5. ~~**Force imbalance (L=175, R=237)**~~: ANSWERED — **Learned behavior**, not structural URDF asymmetry. Symmetry loss will fix it.

6. **Episode length stuck at ~500**: Open. Likely push forces too strong for wide-stance gait. Run 46 comparison with Run 40 termination reasons needed.

## Remaining Open Questions

- **hip_pitch sign in URDF**: Positive left hip_pitch = positive right hip_pitch (same sign)? Needs explicit URDF verification before Run 48.
- **Run 50 curriculum state machine**: Metric definition (raw reward term mean? normalized tracking ratio?), update cadence, persistence location, and logging keys not yet specified.

6. **Episode length stuck at ~500**: Run 40 reached ~1000. What's causing earlier termination in Run 46? Is it the push forces or the wider stance being less stable?
=== END: docs/TRAINING_PLAN.md (latest) ===
=== FILE: docs/ENGINEAI_VS_ISAACLAB.md (latest) ===
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
| Actor input | 252 (42×15 history) | 705 (47×15 history) | **334** (64+18×15 compact history, Run 46+) |
| Critic input | ~345 (115×3 history) | 219 (73×3 history) | 334 (same as actor) |
| Normalizer | Baked into network `forward()` | Not used | Applied externally (RSL-RL EmpiricalNorm) |
| Init | Optional orthogonal (gain=sqrt(2) hidden, 0.01 output) | Default | PyTorch defaults |
| Min std clamp | Yes (per-joint, based on joint limits) | No | No |
| init_noise_std | 1.0 | 1.0 | 1.0 |
| Symmetry loss | No | **Yes** (coef=1.0) | No (planned Run 48) |

**Key differences:**
- rl_workspace_ori actor is smaller [256,256], legged_gym uses [512,256,128]. Both use large inputs (252/705) due to history stacking.
- Run 46 added 15-frame compact history: 334 dims (64 current + 18×15). EngineAI uses 47-dim per frame (705 total) — same design philosophy, our compact 18-dim frames keep 334 < 512 first hidden layer (no bottleneck).
- **History note:** Run 44 failed with 64×15=960-dim (NOT a normalizer issue — 960 > 512 first hidden layer caused compression bottleneck). Run 46 fixed by using compact 18-dim frames.
- legged_gym has symmetry loss enforcing left-right mirror consistency — planned for Run 48.

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

### Our IsaacLab (Run 46+, compact history)

```
Current frame (64-dim):
  lin_vel_b(3) + ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) +
  joint_vel(12) + prev_actions(12) + commands(3) + sin_phase(1) + cos_phase(1) +
  ref_diff(12) + contact_mask(2) = 64

Compact history (270-dim): 15 frames × 18 dims each
  Per frame: ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18
  Buffer: [4096, 15, 18], zero-filled on episode reset

Total: 64 + 270 = 334 dims
```

### Comparison

| Aspect | rl_workspace_ori | legged_gym | Our IsaacLab (Run 46+) |
|--------|-----------------|------------|------------------------|
| Actor obs history | **15 timesteps** | **15 timesteps** | **15 timesteps** ✓ |
| Critic obs history | **3 timesteps** | **3 timesteps** | 0 (planned Run 49) |
| Actor total input dim | **~635** | **705** | **334** |
| Actor per-frame dims | 42 (full obs) | 47 (full obs) | **18 (compact)** |
| Actor sees lin_vel? | **No** (inferred) | **No** (inferred) | **Yes** (current frame) |
| Actor sees projected_gravity? | **No** | **No** | **Yes** (current + history) |
| Actor sees base_euler_xyz? | **Yes** | **Yes** | **No** (uses proj gravity) |
| Critic privileged info? | **Yes** | **Yes** | **No** (planned Run 49) |
| Obs noise | **Yes** (level=1.5) | **Yes** (level=1.5) | **No** (planned Run 49) |
| Obs lag simulation | **Yes** (motor/IMU) | **Yes** (motor/IMU) | **No** (planned Run 49) |
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

**EngineAI QUADRUPED (general):**
```python
rew = sum((air_time - 0.5) * first_contact)   # penalizes < 0.5s
rew *= (norm(cmd[:2]) > 0.1)                   # zero-command gate
```

**EngineAI BIPED (PM01) — rewards_biped.py:**
```python
rew = sum(air_time.clamp(0, 0.5) * first_contact)  # always positive, capped at 0.5s
# NO zero-command gate, NO subtract threshold
```

**Ours (Run 32, matches biped):**
```python
rew = 1.5 * sum(air_time_on_contact.clamp(0, 0.5) * first_contact)
# No velocity gate (matching biped)
```

**Key insight:** Biped formula is COMPLETELY different from quadruped. Biped rewards ANY step
proportionally (always positive), quadruped penalizes short steps. We were using quadruped
formula (Runs 27-31) which made stepping undiscoverable through exploration.

Also: biped includes `stance_mask` in `contact_filt` so air time only accumulates during swing phase.
`contact_filt = contact OR last_contacts OR stance_mask` (biped_robot.py:116-125).

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
| knee_distance | 0.2 | 0.2 | 0.2 | Yes (added Run 39) |
| foot_slip | -0.1 | -0.1 | -0.1 | Yes |
| base_acc | 0.2 | 0.2 | 0.2 | Yes (added Run 39) |
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
- [x] **10. ref_joint_pos formula** — Live code already uses `exp(-2*norm(diff)) - 0.2*clamp(norm,0,0.5)` (EngineAI-style). Verified by Codex review. Hip splay fix reframed: target `default_joint_pos` with per-joint weighting for indices 1,2 (hip_roll_l, hip_yaw_l) and 7,8 (hip_roll_r, hip_yaw_r). **Planned Run 47 Change 1 (reframed).**
- [x] **11. default_joint_pos formula** — updated to EngineAI's `exp(-abs_sum * 100)` in Run 8
- [ ] **12. orientation formula** — EngineAI combines euler angles + projected gravity; we use projected gravity only. **Planned Run 47.** `(exp(-|euler_xy|*10) + exp(-norm(grav_xy)*20)) / 2` gives stronger upright signal, reduces forward lean.
- [ ] **13. feet_clearance formula** — EngineAI scales target by swing_curve (sinusoidal); we use flat target
- [ ] **14. Command curriculum** — start narrow, expand when tracking > 80%. **Planned Run 50.**
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

### Run 32 Biped-Specific Fixes

- [x] **28. Air-time formula: quadruped → biped** (Run 32)
  - Quadruped: `(air_time - 0.5) * first_contact` — penalizes steps < 0.5s
  - Biped (PM01): `clamp(air_time, 0, 0.5) * first_contact` — always positive, capped
  - Source: `engineai_gym/envs/robots/biped/rewards_biped.py:25-28`
  - Runs 27-31 used quadruped formula — made stepping undiscoverable
- [x] **29. contact_filt: add stance_mask** (Run 32, biped-specific)
  - Biped: `contact_filt = contact OR last_contacts OR stance_mask`
  - Ensures air_time only accumulates during swing phase
  - Source: `engineai_gym/envs/robots/biped/biped_robot.py:116-125`
- [x] **30. Remove velocity gate on air-time** (Run 32, biped-specific)
  - Quadruped gates by `vel_cmd > 0.1`, biped does not
  - Source: `rewards_biped.py` has no gate

### Future (sim-to-real transfer)

- [x] **18. Add observation history** — **DONE Run 46**: 15-frame compact history, 334-dim obs. Actor history matches EngineAI. Critic history (3 frames, privileged) planned Run 49.
- [ ] **19. Add asymmetric actor-critic** (privileged critic with forces, friction, mass). **Planned Run 49.**
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
- [ ] **24. Symmetry loss** (legged_gym: sym_coef=1.0, left-right mirror enforcement). **Planned Run 48.** Requires copying RSL-RL PPO into project and adding obs/act permutation matrices. Permutation defined for our 334-dim obs in `docs/TRAINING_PLAN.md`.
- [ ] **25. Terrain curriculum** — 20 levels, 7 terrain types
=== END: docs/ENGINEAI_VS_ISAACLAB.md (latest) ===
=== FILE: armrobotlegging_env_cfg.py (latest) ===
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ArmRobotLegging.robots.pm01 import PM01_CFG


@configclass
class ArmrobotleggingEnvCfg(DirectRLEnvCfg):
    # ---------- timing ----------
    decimation = 4
    episode_length_s = 20.0

    # ---------- spaces ----------
    # single frame obs: base_lin_vel_b(3) + base_ang_vel_b(3) + projected_gravity_b(3) +
    #      joint_pos_rel(12) + joint_vel(12) + prev_actions(12) +
    #      commands(3) + gait_phase(2) + ref_joint_diff(12) + contact_mask(2) = 64
    # Run 46: compact history — 15 frames matching EngineAI frame_stack=15
    #   history signals: ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18 dims/frame
    #   history size: 18 × 15 = 270 extra dims
    #   total obs: 64 + 270 = 334 dims
    # Run 44 failed: full 64×15=960 — input > first hidden layer (512) caused bottleneck
    # Run 45 used 3 frames (118-dim) — safe but limited temporal context
    # Run 46: compact 18-dim × 15 frames = 334 total — no bottleneck (334 < 512), matches EngineAI design
    obs_history_len: int = 15           # Run 46: 15 frames (matches EngineAI frame_stack=15)
    obs_history_size: int = 18          # unchanged: ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12)
    action_space = 12
    observation_space = 64 + 18 * 15   # 334 = 64 current + 270 compact history
    state_space = 0

    # ---------- simulation — 200 Hz physics, 50 Hz policy ----------
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # ---------- terrain ----------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ---------- scene ----------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # ---------- robot ----------
    robot_cfg: ArticulationCfg = PM01_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # leg joints controlled by the policy (order matters — matches action indices)
    leg_joint_names: list = [
        "j00_hip_pitch_l",    # 0
        "j01_hip_roll_l",     # 1
        "j02_hip_yaw_l",      # 2
        "j03_knee_pitch_l",   # 3
        "j04_ankle_pitch_l",  # 4
        "j05_ankle_roll_l",   # 5
        "j06_hip_pitch_r",    # 6
        "j07_hip_roll_r",     # 7
        "j08_hip_yaw_r",      # 8
        "j09_knee_pitch_r",   # 9
        "j10_ankle_pitch_r",  # 10
        "j11_ankle_roll_r",   # 11
    ]

    # bodies used for contact / termination
    foot_body_names: list = ["link_ankle_roll_l", "link_ankle_roll_r"]
    knee_body_names: list = ["link_knee_pitch_l", "link_knee_pitch_r"]
    termination_contact_body_names: list = [
        "link_base",
        # knee/torso have no collision in legs-only simple collision URDF
    ]

    # ---------- action ----------
    action_scale: float = 0.55  # Run 39: 0.6→0.55 — slight reduction for stability (anti-wobble)

    # ---------- gait parameters ----------
    cycle_time: float = 0.8               # gait cycle duration [s] (matching EngineAI)
    target_joint_pos_scale: float = 0.26  # Run 20: revert to EngineAI value (0.17 broke stepping)

    # ---------- velocity command ranges ----------
    cmd_lin_vel_x_range: tuple = (0.3, 1.0)     # [m/s] forward only — eliminates standing-still exploit
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)    # [m/s] reduced lateral range
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)    # [rad/s] reduced yaw range
    cmd_resample_time_s: float = 8.0             # resample commands every N seconds
    cmd_still_ratio: float = 0.0                 # Run 40: no standing — 100% walking focus (standing = fixed motor mode)

    # ---------- domain randomization: push forces ----------
    push_robots: bool = True            # enable random pushes (velocity impulses)
    push_interval_s: float = 8.0        # Run 35: EngineAI value (was 15.0 — more frequent = better reactive stepping)
    max_push_vel_xy: float = 1.0        # Run 33: match EngineAI (was 0.5 — stronger pushes for resilience)
    max_push_ang_vel: float = 0.6       # Run 33: match EngineAI (was 0.4)

    # ---------- domain randomization: PD gains ----------
    pd_gains_rand: bool = True                    # Run 18: randomize stiffness/damping ±20%
    stiffness_multi_range: tuple = (0.8, 1.2)     # multiplier range for stiffness
    damping_multi_range: tuple = (0.8, 1.2)       # multiplier range for damping

    # ---------- termination ----------
    termination_height: float = 0.45    # Run 37: revert (0.65 caused standing still in Run 35+36 — too strict for exploration)
    base_height_target: float = 0.8132  # nominal standing height [m]

    # ---------- contact sensor (Run 31: force-based, replaces z-height) ----------
    # EngineAI uses contact_forces[:, foot_indices, 2] > 5.0 N
    # Z-height contact (threshold=0.16m) was unreliable: foot at 0.148m standing means
    # only 0.012m margin. Shuffling feet at 0.15m stayed "in contact" — no air-time penalty.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,  # update every physics step
        track_air_time=True,
    )
    contact_force_threshold: float = 5.0  # [N] — vertical force above this = contact (EngineAI value)
    contact_height_threshold: float = 0.16  # [m] — kept for legacy, no longer used for foot contact

    # ---------- curriculum: swing penalty annealing (Run 18) ----------
    swing_penalty_start: float = -1.5           # aggressive at start (forces stepping)
    swing_penalty_end: float = -0.8             # relaxed at end (allows survival)
    swing_curriculum_steps: int = 144000        # anneal over ~3000 iters (3000 * 48 steps)

    # ---------- domain randomization: friction ----------
    friction_rand: bool = False                   # Run 42b: DISABLED — disrupted early learning in Run 42 (vel_x stuck ~0.05)
    friction_range: tuple = (0.7, 1.3)            # multiplier range for static+dynamic friction

    # ---------- reward scales (Run 43: exact Run 40 config, reproducibility test) ----------
    # Run 41-42b: ALL changes from Run 40 caused stepping-in-place.
    # Run 43: revert to EXACT Run 40 config to confirm baseline still works.
    #
    # velocity tracking — FULL EngineAI (Run 36: boost to drive forward walking)
    rew_tracking_lin_vel: float = 1.4        # Run 36: FULL EngineAI (was 0.93) — stronger forward drive
    rew_tracking_ang_vel: float = 1.1        # Run 36: FULL EngineAI (was 0.73) — match velocity tracking
    rew_tracking_sigma: float = 5.0          # EngineAI value (not a weight)

    # gait quality — ref_joint_pos reverted, orientation kept full
    rew_ref_joint_pos: float = 1.47          # Run 36: REVERT to /1.5 (2.2 caused standing still in Run 35)
    rew_feet_air_time: float = 1.0           # Run 34: EngineAI 1.5 / 1.5 (keep)
    rew_feet_contact_number: float = 0.93    # Run 34: EngineAI 1.4 / 1.5 (keep)
    rew_orientation: float = 1.0             # Run 35: FULL EngineAI — stronger upright incentive (keep)
    rew_base_height: float = 0.2             # Run 35: FULL EngineAI — maintain standing height (keep)
    rew_feet_clearance: float = -1.6          # Run 38: FULL EngineAI (was -1.07 — caused shuffling)
    rew_default_joint_pos: float = 0.53      # Run 34: EngineAI 0.8 / 1.5 (keep)
    rew_feet_distance: float = 0.13          # Run 34: EngineAI 0.2 / 1.5 (keep)

    # feet distance limits [m]
    min_feet_dist: float = 0.15
    max_feet_dist: float = 0.8
    target_feet_height: float = 0.06         # Run 40: match original robot (3-5cm clearance) — was 0.12
    max_feet_height: float = 0.10            # Run 40: tighter range for natural steps — was 0.18
    rew_feet_height_max: float = -0.6        # Run 38: FULL EngineAI (was -0.4) — penalize too-low feet

    # penalties — stability-critical boosted, rest /1.5
    rew_action_smoothness: float = -0.003    # Run 39: FULL EngineAI (was -0.002 — stronger smoothness for anti-wobble)
    rew_energy: float = -0.000067            # Run 34: EngineAI -0.0001 / 1.5 (keep)
    rew_vel_mismatch: float = 0.5            # Run 35: FULL EngineAI (was 0.33) — penalize unwanted z-motion
    rew_foot_slip: float = -0.067            # Run 34: EngineAI -0.1 / 1.5 (keep)
    rew_alive: float = 0.0                   # not in EngineAI
    rew_termination: float = -0.0            # EngineAI uses -0.0
    rew_track_vel_hard: float = 0.5          # Run 35: FULL EngineAI (was 0.33) — force velocity tracking
    rew_low_speed: float = 0.2               # Run 35: FULL EngineAI (was 0.13) — punish slowness
    rew_dof_vel: float = -6.7e-6             # Run 34: EngineAI -1e-5 / 1.5 (keep)
    rew_dof_acc: float = -3.3e-9             # Run 34: EngineAI -5e-9 / 1.5 (keep)
    rew_lat_vel: float = 0.04                # Run 43: revert to Run 40 value (0.06 may have caused 42b failure)
    rew_swing_phase_ground: float = 0.0      # DISABLED (not in EngineAI)

    # --- Run 39: new anti-wobble rewards from EngineAI ---
    rew_base_acc: float = 0.2               # EngineAI: exp(-norm(base_acc) * 3) — rewards smooth base motion
    rew_knee_distance: float = 0.2          # EngineAI: prevents knee collision, keeps legs spaced
    target_knee_dist: float = 0.25          # target distance between knees [m]

    # --- Run 41/41b: force_balance REMOVED — caused stepping-in-place at any weight ---
    rew_force_balance: float = 0.0          # Run 42: DISABLED (0.08 and 0.15 both caused stepping-in-place)
=== END: armrobotlegging_env_cfg.py (latest) ===

Now answer this review question:

We just implemented Run 47. Review the two changes in armrobotlegging_env.py: (1) orientation formula at line ~757 — is the dual-signal implementation correct? (2) default_joint_pos at line ~805 — are the splay indices [1,2,7,8] correct and is the norm+deadband formula right? Any bugs or issues before we run training?

mcp startup: no servers
codex
I can review the intended Run 47 patch logic, but I cannot verify the exact implementation text because the updated [`armrobotlegging_env.py`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py) was not included in this message. Based on the two target changes, these are the checks that matter before training.

**Findings**

1. **Potential bug at the new orientation block around [`armrobotlegging_env.py:757`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py#L757): `euler_xyz_from_quat` return shape.**  
   If the implementation does something like:
   ```python
   euler_xyz = euler_xyz_from_quat(base_quat)
   quat_mismatch = torch.exp(-torch.sum(torch.abs(euler_xyz[:, :2]), dim=1) * 10.0)
   ```
   that is likely wrong. In IsaacLab this helper is typically tuple-like, so the safe form is:
   ```python
   roll, pitch, _ = euler_xyz_from_quat(base_quat)
   base_euler_xy = torch.stack((roll, pitch), dim=-1)
   ```
   The EngineAI target formula from [`zqsa01.py:293-295`](/home/armmarov/work/robot/engineai/engineai_legged_gym/legged_gym/envs/zq_sa_01/zqsa01.py#L293) is otherwise correct.

2. **The splay indices `[1, 2, 7, 8]` are correct, but the formula is only correct if it uses pairwise left/right norms plus one deadbanded scalar, not a flat 4D norm or per-joint deadbands.**  
   For Change 1 at around [`armrobotlegging_env.py:805`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py#L805), the target joints are right:
   - left hip splay: `1, 2`
   - right hip splay: `7, 8`

   The preferred formula is:
   ```python
   left_splay = joint_pos_rel[:, [1, 2]]
   right_splay = joint_pos_rel[:, [7, 8]]
   splay_dev = torch.norm(left_splay, dim=1) + torch.norm(right_splay, dim=1)
   splay_dev = torch.clamp(splay_dev - 0.1, 0.0, 50.0)
   rew_default_pos = cfg.rew_default_joint_pos * (
       torch.exp(-splay_dev * 100.0) - 0.01 * torch.norm(joint_pos_rel, dim=1)
   )
   ```
   If the implementation instead does `torch.norm(joint_pos_rel[:, [1,2,7,8]], dim=1)` that is still usable, but it is less faithful to the EngineAI left/right structure.

**Answers**

1. **Orientation dual-signal implementation**
   - Conceptually correct if it now does:
     - `quat_mismatch = exp(-sum(abs(euler_xy)) * 10)`
     - `orientation = exp(-norm(projected_gravity_xy) * 20)`
     - average them and multiply by `cfg.rew_orientation`
   - Keep `rew_orientation=1.0`; that part is correct.
   - Main bug to check: unpacking `euler_xyz_from_quat` correctly.

2. **`default_joint_pos` splay indices and formula**
   - Indices `[1,2,7,8]` are correct for hip roll + hip yaw splay.
   - The formula is right if it uses left/right pair norms and a single `0.1` deadband before the exponential term.
   - Keep the `-0.01 * torch.norm(joint_pos_rel, dim=1)` global term; don’t drop it.

**Before training**
- Verify the orientation code is not indexing directly into the raw return of `euler_xyz_from_quat`.
- Verify hip pitch indices `0,6` are no longer in the splay block.
- Verify the deadband is applied to the combined splay magnitude, not independently per joint unless that was a deliberate design change.
- Expect the orientation reward term to numerically drop at first because it is stricter; that alone is not a regression.

If you paste the two updated code blocks from [`armrobotlegging_env.py`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py), I can do an exact pass and tell you whether there is any implementation bug.
tokens used
237,801
I can review the intended Run 47 patch logic, but I cannot verify the exact implementation text because the updated [`armrobotlegging_env.py`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py) was not included in this message. Based on the two target changes, these are the checks that matter before training.

**Findings**

1. **Potential bug at the new orientation block around [`armrobotlegging_env.py:757`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py#L757): `euler_xyz_from_quat` return shape.**  
   If the implementation does something like:
   ```python
   euler_xyz = euler_xyz_from_quat(base_quat)
   quat_mismatch = torch.exp(-torch.sum(torch.abs(euler_xyz[:, :2]), dim=1) * 10.0)
   ```
   that is likely wrong. In IsaacLab this helper is typically tuple-like, so the safe form is:
   ```python
   roll, pitch, _ = euler_xyz_from_quat(base_quat)
   base_euler_xy = torch.stack((roll, pitch), dim=-1)
   ```
   The EngineAI target formula from [`zqsa01.py:293-295`](/home/armmarov/work/robot/engineai/engineai_legged_gym/legged_gym/envs/zq_sa_01/zqsa01.py#L293) is otherwise correct.

2. **The splay indices `[1, 2, 7, 8]` are correct, but the formula is only correct if it uses pairwise left/right norms plus one deadbanded scalar, not a flat 4D norm or per-joint deadbands.**  
   For Change 1 at around [`armrobotlegging_env.py:805`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py#L805), the target joints are right:
   - left hip splay: `1, 2`
   - right hip splay: `7, 8`

   The preferred formula is:
   ```python
   left_splay = joint_pos_rel[:, [1, 2]]
   right_splay = joint_pos_rel[:, [7, 8]]
   splay_dev = torch.norm(left_splay, dim=1) + torch.norm(right_splay, dim=1)
   splay_dev = torch.clamp(splay_dev - 0.1, 0.0, 50.0)
   rew_default_pos = cfg.rew_default_joint_pos * (
       torch.exp(-splay_dev * 100.0) - 0.01 * torch.norm(joint_pos_rel, dim=1)
   )
   ```
   If the implementation instead does `torch.norm(joint_pos_rel[:, [1,2,7,8]], dim=1)` that is still usable, but it is less faithful to the EngineAI left/right structure.

**Answers**

1. **Orientation dual-signal implementation**
   - Conceptually correct if it now does:
     - `quat_mismatch = exp(-sum(abs(euler_xy)) * 10)`
     - `orientation = exp(-norm(projected_gravity_xy) * 20)`
     - average them and multiply by `cfg.rew_orientation`
   - Keep `rew_orientation=1.0`; that part is correct.
   - Main bug to check: unpacking `euler_xyz_from_quat` correctly.

2. **`default_joint_pos` splay indices and formula**
   - Indices `[1,2,7,8]` are correct for hip roll + hip yaw splay.
   - The formula is right if it uses left/right pair norms and a single `0.1` deadband before the exponential term.
   - Keep the `-0.01 * torch.norm(joint_pos_rel, dim=1)` global term; don’t drop it.

**Before training**
- Verify the orientation code is not indexing directly into the raw return of `euler_xyz_from_quat`.
- Verify hip pitch indices `0,6` are no longer in the splay block.
- Verify the deadband is applied to the combined splay magnitude, not independently per joint unless that was a deliberate design change.
- Expect the orientation reward term to numerically drop at first because it is stricter; that alone is not a regression.

If you paste the two updated code blocks from [`armrobotlegging_env.py`](/home/armmarov/work/robot/isaac/workspace/armrobot/ArmRobotLegging/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py), I can do an exact pass and tell you whether there is any implementation bug.
