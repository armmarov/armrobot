# PM01 Walking — Training Plan

This document tracks planned improvements, their motivation, implementation details, and review status.
Designed for multi-model review — each plan item should be independently evaluable.

---

## Current State (Run 47 — killed iter ~1306)

| Item | Value |
|------|-------|
| Obs space | 334-dim (64 current + 15×18 compact history) |
| vel_x | ~0.05 m/s — standing-still exploit |
| Episode length | 999 (between spikes) |
| noise_std | 0.53 (at kill) |
| Key weakness | (1) PPO value loss spikes 10k–15k → policy collapse; (2) robot stands still, doesn't walk |

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

## Run 48 — PPO Stability + Standing-Still Fix

**Goal:** Fix two root causes found in Run 47 before proceeding to symmetry loss. Config-only changes — no code modification.

**Status:** Planned. Two changes, both in config files only.

---

### Change 1 — Disable value loss clipping (`rsl_rl_ppo_cfg.py`)

**Problem:** RSL-RL clips value function updates by `±clip_param = ±0.2`. Our discounted returns are ~500–600. The value function cannot move more than 0.2 per update step, so it can never catch up when the policy improves suddenly. The raw MSE loss is reported at 10,000–14,000 (the gap between frozen estimates and real returns). Bad advantage estimates → bad policy updates → reward collapse every ~250 iters.

**Fix:**
```python
use_clipped_value_loss=False,  # was True — ±0.2 clip is too tight at return scale ~500
```

`max_grad_norm=1.0` is still active and bounds the gradient norm, providing safety without the broken scale mismatch.

**Risk:** Low. Value function was barely updating before. This lets it learn properly.

---

### Change 2 — Tighten velocity tracking sigma (`armrobotlegging_env_cfg.py`)

**Problem:** `rew_tracking_sigma=5.0` is too lenient. With sigma=5.0, a robot standing still (vel_error=0.3 m/s) gets tracking reward = 1.375 vs 1.400 for walking — a difference of only 0.025. Orientation+posture rewards dominate and the robot never learns to walk. vel_x stayed below 0.056 the entire Run 47.

**Fix:**
```python
rew_tracking_sigma: float = 1.0  # was 5.0 — standing still at 0.3 m/s now costs 0.122 (5× larger signal)
```

| sigma | Reward standing still (cmd=0.3) | Reward walking | Difference |
|-------|--------------------------------|----------------|------------|
| 5.0 (current) | 1.375 | 1.400 | **0.025** (too small) |
| 1.0 (proposed) | 1.278 | 1.400 | **0.122** (5× larger) |
| 0.25 (Run 1, too sharp) | 0.914 | 1.400 | 0.486 (too punishing early) |

**Note:** The Key Lessons warn "sigma=0.25 too sharp, use 5.0" (Run 1). But Run 1 failed because the policy couldn't balance at all — not because sigma was wrong. Run 47 robot CAN balance (ep_len=999). sigma=1.0 is the right middle ground.

**Risk:** Low-medium. May slow early learning if robot falls more. Watch ep_len in first 500 iters — if it drops below 200 consistently, consider sigma=2.0 as fallback.

---

## Run 49 — Symmetry Loss (was Run 48)

**Goal:** Enforce L/R mirror symmetry at the PPO level — the main EngineAI feature we're missing.

**Status:** Planned after Run 48 confirms stable walking.

> *(Was Run 48 — shifted one run due to stability fixes needed first)*

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

## Run 50 — Domain Randomization + Observation Noise (was Run 49)

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

## Run 51 — Command Curriculum (was Run 50)

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
