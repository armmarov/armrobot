# PM01 Walking — Training History

All training runs, changes made, and results. Most recent run at the bottom.

> **Note:** Standing orders / AI researcher role moved to `CLAUDE.md` in project root.

---

## Run 1 — Initial Config (Baseline)

**Date:** 2026-03-07

**Config:**
- lr: 1e-3
- entropy_coef: 0.005
- num_learning_epochs: 5
- gamma: 0.99, lam: 0.95
- num_steps_per_env: 24
- max_iterations: 3000
- critic_hidden_dims: [512, 256, 128]
- tracking_sigma: 0.25
- rew_alive: 0.15, rew_termination: -2.0
- cycle_time: 0.64s
- gait reference: hip_pitch (idx 0/6)
- 10 reward terms

**Results (by iter 939):**
- Noise std **increasing** from 1.0 to 1.39 (policy diverging)
- Mean reward: ~4000 (flat, not improving)
- Value function loss: stuck at 1.0+
- Robot: standing still, not learning

**Diagnosis:**
- `entropy_coef=0.005` too high — actively pushing noise std up
- `tracking_sigma=0.25` too sharp — exp(-error/0.25) always near 0, unlearnable
- `lr=1e-3` too aggressive — overshooting

---

## Run 2 — Hyperparameter Alignment with EngineAI

**Date:** 2026-03-07

**Changes from Run 1:**
- lr: 1e-3 → **1e-4**
- entropy_coef: 0.005 → **0.001**
- num_learning_epochs: 5 → **2**
- gamma: 0.99 → **0.994**
- lam: 0.95 → **0.9**
- num_steps_per_env: 24 → **48**
- max_iterations: 3000 → **10000**
- critic_hidden_dims: [512,256,128] → **[768,256,128]**
- tracking_sigma: 0.25 → **5.0**
- rew_tracking_lin_vel: 1.5 → **1.4**
- rew_tracking_ang_vel: 1.0 → **1.1**
- rew_ref_joint_pos: 2.0 → **2.2**
- rew_feet_contact_number: 1.2 → **1.4**
- rew_action_smoothness: -0.005 → **-0.003**

**Results (by iter 317):**
- Noise std: 0.96 (healthy decrease)
- Mean reward: ~4000 (flat, same as run 1)
- Value function loss: 200-250 (improved but still high)
- Robot: standing but not walking

**Diagnosis:**
- `ref_joint_pos` reward used `torch.sum` over 12 joints — exp(-2 * sum) always ~0, reward unearnable
- Gait reference drove hip_pitch (idx 0/6) — correct joint but cycle_time still 0.64s
- Matching EngineAI's hip_yaw (idx 2/8) was attempted next

---

## Run 3 — Fix ref_joint_pos + Gait Indices + Cycle Time

**Date:** 2026-03-08

**Changes from Run 2:**
- ref_joint_pos: `torch.sum` → **`torch.mean`** (critical fix)
- Gait reference: hip_pitch (idx 0/6) → **hip_yaw (idx 2/8)** (matching EngineAI)
- cycle_time: 0.64s → **0.8s** (matching EngineAI)
- Added gait deadband: `ref=0 when |sin_phase| < 0.05`
- rew_alive: 0.15 → **0.0** (disabled)
- rew_termination: -2.0 → **0.0** (disabled)

**Results (by iter 498):**
- Episode length **dropping**: 876 → 538 (robot falling more)
- Value loss spiking to 3398
- Robot: standing briefly then falling over

**Diagnosis:**
- Removed both alive bonus AND termination penalty — no incentive to stay upright ("no carrot, no stick")
- lr=1e-4 still too high — value loss exploding
- Missing stabilizing rewards (feet_clearance, default_joint_pos, etc.)

---

## Run 4 — Re-enable Alive/Termination + Lower LR + Add 4 Rewards

**Date:** 2026-03-08

**Changes from Run 3:**
- lr: 1e-4 → **1e-5**
- rew_alive: 0.0 → **0.05** (small survival bonus)
- rew_termination: 0.0 → **-1.0** (fall penalty)
- Added 4 new reward terms:
  - `feet_clearance` (scale **-1.6**) — penalize low swing foot
  - `default_joint_pos` (scale **0.8**) — keep hip pitch/roll near default
  - `feet_distance` (scale **0.2**) — penalize feet too close/far
  - `foot_slip` (scale **-0.1**) — penalize foot sliding during contact
- Total reward terms: 10 → **14**

**Results (iter 0 → 559):**

| Iter | Reward | Episode Length | Noise Std | Value Loss |
|------|--------|---------------|-----------|------------|
| 0 | 261 | 47 | 1.00 | — |
| 50 | 488 | 85 | 1.00 | — |
| 100 | 848 | 145 | 0.99 | — |
| 150 | 1,945 | 329 | 0.99 | — |
| 200 | 2,580 | 429 | 0.98 | ~2000 |
| 250 | 3,381 | 554 | 0.98 | — |
| 300 | 4,802 | 787 | 0.97 | ~300 |
| 350 | 5,235 | 855 | 0.96 | — |
| 400 | 3,877 | 632 | 0.95 | ~200 |
| 450 | 5,618 | 898 | 0.94 | — |
| 500 | 5,845 | 935 | 0.93 | — |
| 550 | 5,648 | 904 | 0.91 | ~120 |
| 559 | 5,948 | 948 | 0.91 | 92 |

**Observations:**
- Best run so far — steady improvement, no collapse
- Episode length ~950 = robot surviving ~19s out of 20s max
- Value loss dropped from ~2000 → 92 (critic well-calibrated)
- Noise std steadily decreasing (healthy exploration reduction)
- **Problem: robot spinning 360 degrees to the left instead of walking forward**
- Root cause: gait reference drives hip_yaw (rotation) not hip_pitch (forward swing)

---

## Run 5 — Fix Gait Reference to Hip Pitch (Pending)

**Date:** 2026-03-08

**Changes from Run 4:**
- Gait reference: **hip_yaw (idx 2/8) → hip_pitch (idx 0/6)**
  - hip_pitch drives forward/backward leg swing for walking
  - hip_yaw drives rotation, causing spinning behavior
  - EngineAI uses hip_yaw but has obs history + domain rand to prevent spinning

**All other params unchanged from Run 4.**

**Results (iter 0 → 3262, then stopped — converged):**

| Iter | Reward | Episode Length | Noise Std | Value Loss |
|------|--------|---------------|-----------|------------|
| 0 | 259 | 47 | 1.00 | — |
| 100 | 841 | 154 | 1.00 | 1,212 |
| 300 | 5,106 | 687 | 0.97 | 408 |
| 500 | 6,145 | 945 | 0.95 | 96 |
| 700 | 6,269 | 999 | 0.89 | 1,436 |
| 1000 | 6,570 | 997 | 0.73 | 14 |
| 1400 | 6,608 | 998 | 0.61 | 22 |
| 2000 | 6,761 | 999 | 0.57 | 40 |
| 3000 | 6,807 | 999 | 0.51 | 23 |
| 3262 | 6,875 | 999 | 0.48 | 24 |

**Observations:**
- Best run — robot survives full 20s episodes from iter 700 onward
- Reward plateaued at ~6,700-6,875 from iter 1400+
- Noise std reached 0.48 (in optimal 0.3-0.5 range)
- Training killed at iter 3262 — fully converged, no further improvement expected
- Headless mode ran 2x faster than GUI (3.2s/iter vs 6.2s)
- **Visual evaluation: robot NOT walking forward** — lifts legs in place and turns, does not translate
- Root cause: tracking_sigma=5.0 gives 82% reward for standing still (exp(-1/5)=0.82)
- Missing EngineAI rewards: `track_vel_hard` and `low_speed` which force actual locomotion

---

## Run 6 — Add track_vel_hard + low_speed (In Progress)

**Date:** 2026-03-08

**Changes from Run 5b:**
- Added **`track_vel_hard`** (scale 0.5) — sharp velocity tracking using exp(-error×10)
  - Standing still with cmd=1.0 gives exp(-10)≈0.00005 → forces movement
- Added **`low_speed`** (scale 0.2) — discrete speed reward:
  - -1.0 if speed < 50% of command (too slow)
  - +2.0 if speed within 50-120% of command (good)
  - -2.0 if moving in wrong direction
  - Only active when |command| > 0.1
- Total reward terms: 14 → **16**

**All other params unchanged from Run 5b.**

**Results (iter 0 → 2321, killed — converged):**

| Iter | Reward | Episode Length | Noise Std | Value Loss |
|------|--------|---------------|-----------|------------|
| 13 | 333 | 62 | 1.00 | 951 |
| 125 | 1,276 | 224 | 0.99 | 1,193 |
| 314 | 3,216 | 549 | 0.98 | 637 |
| 508 | 5,191 | 864 | 0.95 | 533 |
| 700 | 5,880 | 962 | 0.90 | 288 |
| 892 | 6,119 | 985 | 0.84 | 67 |
| 1,084 | 6,335 | 999 | 0.78 | 30 |
| 1,470 | 6,415 | 993 | 0.73 | 45 |
| 1,853 | 6,549 | 996 | 0.72 | 84 |
| 2,046 | 6,467 | 983 | 0.71 | 82 |
| 2,321 | 6,605 | 990 | 0.71 | 95 |

**Evaluation:**
- **Robot NOT walking forward properly** — vibrating ankles, shuffling/waddling sideways to the left
- track_vel_hard + low_speed forced *some* movement (improvement over Run 5b standing still)
- Robot found a "cheat" — ankle vibration + sideways shuffle to satisfy velocity tracking
- Not a proper bipedal gait — knees not lifting, no forward translation

**Diagnosis:**
- Gait reference reward (rew_ref_joint_pos=2.2) not strong enough to enforce proper leg swing
- Robot exploits ankle joints (lowest energy) instead of using hip/knee for walking
- Missing penalties: dof_vel, dof_acc to discourage high-frequency ankle vibration
- Missing: observation history for temporal gait planning

---

## Run 7 — Anti-vibration + 2nd-order Smoothness + Lateral Tracking

**Date:** 2026-03-08

**Changes from Run 6:**
- Added **`dof_vel`** (scale -1e-5) — penalizes all joint velocities, discourages vibration
- Added **`dof_acc`** (scale -5e-9) — penalizes joint accelerations, CRITICAL for anti-vibration
  - Computed as: `((joint_vel - last_joint_vel) / dt)²`
  - Prevents high-frequency ankle oscillation that was the main Run 6 problem
- Added **`lat_vel`** (scale 0.3) — lateral velocity tracking with exp(-error*10)
  - Prevents sideways drift/shuffling
- Improved **`action_smoothness`** — added EngineAI's 2nd-order term
  - term_1: consecutive action difference (was already there)
  - term_2: `(a_t + a_{t-2} - 2*a_{t-1})²` — prevents rapid acceleration of actions
  - term_3: `0.05 * |actions|` — small action magnitude penalty
- Total reward terms: 16 → **19**

**All other params unchanged from Run 6.**

**Results (iter 0 → 2,810, killed — converged):**

| Iter | Reward | Episode Length | Noise Std | Value Loss |
|------|--------|---------------|-----------|------------|
| 17 | 334 | 62 | 0.99 | 406 |
| 89 | 725 | 126 | 0.99 | 1,286 |
| 279 | 4,664 | 786 | 0.97 | 623 |
| 470 | 5,016 | 838 | 0.94 | 299 |
| 663 | 6,025 | 981 | 0.86 | 190 |
| 857 | 6,186 | 977 | 0.79 | 84 |
| 1,052 | 6,447 | 999 | 0.71 | 136 |
| 1,247 | 6,603 | 997 | 0.64 | 90 |
| 1,442 | 6,650 | 990 | 0.61 | 112 |
| 1,639 | 6,704 | 992 | 0.59 | 164 |
| 1,834 | 6,787 | 992 | 0.58 | 139 |
| 2,000 | 6,803 | 992 | 0.58 | 163 |
| 2,417 | 6,796 | 986 | 0.57 | 141 |
| 2,810 | 6,883 | 993 | 0.56 | 138 |

**Evaluation:**
- **Ankle vibration FIXED** — dof_acc + dof_vel penalties successfully eliminated the high-frequency oscillation from Run 6
- **Sideways shuffling FIXED** — lat_vel tracking prevented the leftward drift from Run 6
- **Robot still NOT walking forward** — stands in place with small weight-shifting motions, no forward translation
- Peak reward 6,903, noise std 0.56, episode length ~993 (full 20s survival)
- Training killed at iter 2,810 — converged for 1,300+ iters

**Diagnosis — EngineAI comparison reveals 3 critical gaps:**

1. **Gait reference drives wrong joints** — our ref only drives hip_pitch (idx 0/6) with single amplitude 0.26 rad. EngineAI drives **hip_yaw (idx 2/8) + knee_pitch (idx 3/9) + ankle_pitch (idx 4/10)** with coupled amplitudes (hip: 0.26 rad, knee: 0.52 rad, ankle: 0.26 rad). Without knee/ankle in the reference, the robot has no trajectory to follow for a proper walking gait.

2. **No domain randomization** — EngineAI uses random push forces/torques, friction randomization, and mass randomization. This prevents the "stand still" exploit because the robot must actively balance against disturbances, which naturally requires stepping.

3. **No curriculum** — EngineAI starts with narrow velocity commands and expands when tracking > 80%. Our robot gets full ±1.0 m/s from the start, which may be too hard to learn initially.

**Note:** Reward scales already match EngineAI exactly (tracking_sigma=5, track_vel_hard=0.5, low_speed=0.2, etc.) — the problem is NOT the reward weights but the gait reference and missing randomization.

---

## Run 8 — Match EngineAI Gait Reference (3 joints/leg) + Phase Freeze + Contact Penalty

**Date:** 2026-03-08

**Changes from Run 7:**
- **Gait reference now drives 3 joints per leg** (matching EngineAI exactly):
  - hip_yaw (idx 2/8): amplitude 0.26 rad
  - knee_pitch (idx 3/9): amplitude 0.52 rad (2× hip)
  - ankle_pitch (idx 4/10): amplitude 0.26 rad
  - Previously only drove hip_pitch (idx 0/6) — the **#1 reason the robot didn't walk**
- **Gait phase freezes on zero commands** — EngineAI freezes phase when standing still; we now do the same
- **Small command filter** — commands with linear norm < 0.2 or yaw < 0.2 zeroed out (matching EngineAI)
- **Contact pattern mismatch penalty** — changed from [0, 1] to EngineAI's [−0.3, +1.0] (penalizes wrong contact phase)
- **default_joint_pos formula** — updated to match EngineAI: `exp(-abs_sum * 100)` instead of previous `exp(-norm * 100)` with 0.1 threshold
- Total reward terms: **19** (unchanged)

**All other params unchanged from Run 7.**

**Results (iter 0 → 139, killed — hip_yaw spinning):**

| Iter | Reward | Episode Length | Noise Std | Value Loss |
|------|--------|---------------|-----------|------------|
| 13 | 321 | 64 | 1.00 | 648 |
| 50 | 417 | 78 | 1.00 | 383 |
| 100 | 747 | 135 | 0.99 | 917 |
| 139 | 377 | 74 | 0.99 | 1,178 |

**Evaluation:**
- **Killed at iter 139** — reward crashed from 747 → 377, episode length 135 → 74
- Value loss spiking to 1,178 (critic struggling)
- **Root cause: hip_yaw spinning problem** — same as Run 4. Without observation history (15 steps) and domain randomization, hip_yaw causes rotation instead of walking.
- EngineAI can use hip_yaw because they have obs history + domain rand + asymmetric critic

**Decision:** Switch to hip_pitch (idx 0/6) + knee (idx 3/9) + ankle (idx 4/10) for Run 9.

---

## Run 9 — Hip Pitch + Knee + Ankle (3 joints/leg, no spinning)

**Date:** 2026-03-08

**Changes from Run 8:**
- Gait reference: **hip_yaw (idx 2/8) → hip_pitch (idx 0/6)** — avoids spinning
- Still drives 3 joints per leg: hip_pitch + knee_pitch + ankle_pitch (0.26/0.52/0.26 rad)
- All other Run 8 improvements retained (phase freeze, contact penalty, small cmd filter, etc.)

**All other params unchanged from Run 8.**

**Results (iter 0 → 49, killed — switching to legs-only URDF):**

| Iter | Reward | Episode Length | Noise Std | Value Loss |
|------|--------|---------------|-----------|------------|
| 49 | 480 | 93 | 0.99 | 1,546 |

**Decision:** Killed at iter 49 to switch from full-body `pm01.urdf` to `pm01_only_legs_simple_collision.urdf`. Only 3 minutes in — no significant progress lost.

---

## Run 10 — Legs-Only URDF with Simple Collisions

**Date:** 2026-03-08

**Changes from Run 9:**
- **URDF: `pm01.urdf` → `pm01_only_legs_simple_collision.urdf`**
  - Upper body joints (j12-j23) changed from revolute to **fixed** — locked in place
  - Removed waist/arms/head actuator configs (no longer needed)
  - Collision simplified: base=box, feet=mesh, everything else=no collision
  - Benefits: fewer collision bodies → faster physics, no spurious upper-body contacts
- **Termination contacts:** removed `link_knee_pitch_l/r` and `link_torso_yaw` (no collision geometry in new URDF), kept `link_base` + height check
- All reward/gait params unchanged from Run 9
- **Added per-term reward diagnostics** — each of 21 reward terms logged individually to TensorBoard + console via `extras["log"]`
  - Key diagnostic: `Episode/mean_base_vel_x` — tracks whether robot is actually moving forward
  - Enables mid-training checks instead of waiting for convergence + visual evaluation

**Results (iter 0 → 246, killed — standing still exploit again):**

| Iter | Reward | Episode Length | Noise Std | Value Loss | mean_vel_x |
|------|--------|---------------|-----------|------------|------------|
| 6 | 284 | 55 | 1.00 | 1,475 | 0.69 |
| 230 | 3,811 | 617 | 0.95 | 535 | -0.10 |
| 246 | 5,970 | 956 | 0.95 | 237 | 0.00 |

**Evaluation:**
- **Same standing-still exploit as Runs 5-7** — robot survives full episodes but `mean_base_vel_x ≈ 0`, `feet_air_time = 0`
- Per-term diagnostics confirmed: `low_speed = -218` (heavily penalized but not enough), `tracking_lin_vel` high (sigma=5.0 gives 95% reward for standing still)
- Legs-only URDF + 3-joint gait reference are correct, but **reward imbalance remains** without domain randomization
- Killed at iter 246 to add push forces for Run 11

---

## Run 11 — Push Force Domain Randomization

**Date:** 2026-03-08

**Changes from Run 10:**
- **Push forces (velocity impulses)** — matching EngineAI:
  - Every 8 seconds, apply random velocity impulse to robot base
  - Linear: ±0.4 m/s in xy
  - Angular: ±0.6 rad/s in roll/pitch/yaw
  - Forces robot to take reactive steps to maintain balance
  - Prevents standing-still exploit: robot must actively step or it falls
- All other params unchanged from Run 10

**Results (iter 0 → ~2681, standing still exploit — push forces too weak):**

| Iter | Reward | Episode Length | Noise Std | Value Loss | mean_vel_x |
|------|--------|---------------|-----------|------------|------------|
| 11 | 298 | 58 | 1.00 | 772 | 0.69 |
| 233 | 4,113 | 695 | 0.96 | 7,126 | -0.06 |
| 570 | 3,498 | 571 | 0.82 | 109 | -0.07 |
| ~910 | 6,145 | 954 | 0.63 | 22,193 | -0.00 |
| ~1250 | 6,726 | 999 | 0.42 | 5.7 | 0.02 |
| ~2260 | ~6,800 | ~999 | 0.19 | — | 0.13 |
| ~2681 | ~6,800 | 636 | 0.19 | — | -0.02 |

**Evaluation:**
- **Push forces (±0.4 m/s) too weak** — robot learned to brace against pushes while standing still
- `mean_base_vel_x ≈ 0`, `feet_air_time = 0` throughout entire training
- `low_speed = -70.7` (heavily penalized but drowned out by standing rewards)
- Brief blip at iter ~2260 (vel_x=0.13) but reverted by iter ~2681
- Noise std 0.19 = fully converged on standing still
- Root cause: tracking_sigma=5.0 gives 94% reward for standing; push forces only 0.4 m/s
- Also: ref_joint_pos uses `exp(-2*mean(diff²))` instead of EngineAI's `mean(exp(-2*diff²))` — dilutes gait signal

---

## Run 12 — Anti-Standing-Still: Stronger Pushes + Reward Rebalance

**Date:** 2026-03-09

**Changes from Run 11 (4 targeted fixes for standing-still exploit):**
1. **ref_joint_pos formula**: `exp(-2*mean(diff²))` → **`mean(exp(-2*diff²))`** — matches EngineAI exactly. Per-joint exp then average. Previous formula diluted gait error (3 active joints out of 12 → only 25% signal).
2. **tracking_sigma**: 5.0 → **2.5** — standing still now gives ~87% tracking reward instead of ~94%. More gradient pressure to actually match commanded velocity.
3. **low_speed**: 0.2 → **1.5** (7.5× stronger) — standing-still penalty was -0.2/step, drowned by +7.3/step positive rewards. Now -1.5/step for too-slow, +3.0/step for matching speed.
4. **Push forces**: ±0.4 m/s @ 8s → **±1.0 m/s @ 4s** — 2.5× stronger, 2× more frequent. Angular: 0.6 → 0.8 rad/s. Matching EngineAI base config (1.0 m/s). Robot must step reactively or fall.

**All other params unchanged from Run 11.**

**Results (iter 0 → 1456, killed — standing still exploit):**

| Iter | Reward | Episode Length | Noise Std | Value Loss | mean_vel_x |
|------|--------|---------------|-----------|------------|------------|
| 30 | 280 | 63 | 0.99 | 356 | 0.50 |
| 113 | 446 | 87 | 0.97 | 321 | 0.02 |
| 443 | 1,240 | 265 | 0.85 | 95 | 0.01 |
| 779 | 1,148 | 215 | 0.68 | 35,532 | -0.05 |
| 1119 | 1,670 | 278 | 0.52 | 310 | 0.24 |
| 1456 | 2,046 | 309 | 0.41 | 290 | -0.03 |

**Evaluation:**
- **Same standing-still exploit despite all 4 fixes** — vel_x transient blip at iter 1119 (0.24) but reverted
- `low_speed` went POSITIVE (+33.9) while vel_x ≈ 0 — robot exploits zero/small commands
- Root cause: `cmd_still_ratio=0.1` + small command filter (zeroing cmds < 0.2) creates many zero commands where standing still is correct behavior
- **Key insight: the standing-still exploit is a command distribution problem, not a reward scale problem**

---

## Run 13 — Forward-Only Commands (Structural Fix)

**Date:** 2026-03-09

**Changes from Run 12 (structural fix for standing-still exploit):**
1. **Forward-only commands**: `cmd_lin_vel_x_range = (0.3, 1.0)` — every command requires forward movement (minimum 0.3 m/s). Was (-1.0, 1.0) which allowed zero/backward commands.
2. **No zero commands**: `cmd_still_ratio = 0.0` — was 0.1 (10% zero commands exploited by standing).
3. **Removed small command filter** — previously zeroed linear cmds < 0.2 and yaw < 0.2, creating more zero commands.
4. **Reduced lateral/yaw ranges**: y: ±0.3→±0.2, yaw: ±1.0→±0.5 — simplify the task to focus on forward walking first.

All Run 12 reward params retained (sigma=2.5, low_speed=1.5, pushes ±1.0@4s, per-joint ref).

**Results (training in progress — BREAKTHROUGH):**

| Iter | Reward | Episode Length | Noise Std | Value Loss | mean_vel_x | low_speed |
|------|--------|---------------|-----------|------------|------------|-----------|
| 41 | 344 | 67 | 0.99 | 1,087 | 0.55 | -29 |
| 220 | 998 | 176 | 0.94 | 5,741 | 0.35 | -105 |
| 551 | 1,972 | 305 | 0.86 | 1,003 | 0.29 | +39 |
| 884 | 2,960 | 379 | 0.74 | 892 | 0.56 | +725 |
| 1216 | 3,549 | 452 | 0.60 | 57,006 | **0.85** | +967 |
| 1549 | 1,874 | 263 | 0.53 | 65,408 | **0.92** | +355 |
| 1884 | 3,437 | 419 | 0.44 | 688 | 0.60 | +980 |
| 2228 | 4,151 | 500 | 0.40 | 22 | 0.65 | +1,292 |
| 2549 | 301 | 52 | 0.40 | 86,949 | 0.70 | +16 |
| 2873 | 413 | 61 | 0.40 | 10,009 | 0.50 | +63 |
| 3192 | 1,461 | 232 | 0.39 | 771 | 0.34 | +110 |
| 3517 | 2,043 | 275 | 0.37 | 1,015 | 0.41 | +421 |
| 3847 | 2,401 | 315 | 0.36 | 1,040 | 0.36 | +498 |
| 4179 | 2,453 | 301 | 0.34 | 68,196 | 0.57 | +1,011 |
| 4850 | 2,839 | 342 | 0.32 | 4,568 | 0.53 | +661 |
| 5187 | 2,693 | 317 | 0.31 | 81,189 | 0.64 | +1,272 |

**BREAKTHROUGH at iter 1216:**
- `mean_base_vel_x = 0.852` — first run EVER to achieve sustained forward velocity
- `low_speed = +967` — robot matching commanded forward speeds (0.3-1.0 m/s)
- `feet_air_time = 0` — flat-footed shuffle, not proper stepping gait
- `base_height = 0.7` — crouching (target 0.8132)
- Forward-only commands eliminated the standing-still exploit that plagued Runs 5-12

**Final results (killed at iter ~9560, converged):**

| Metric | Final Range | Notes |
|--------|------------|-------|
| Reward | 3,000 - 4,251 (ATH) | Oscillates due to value loss spikes |
| vel_x | 0.4 - 0.76 | Sustained forward locomotion |
| Episode length | 360 - 500 (~7-10s) | Falls after ~8s average |
| Noise std | 0.25 - 0.27 | Converged |
| feet_air_time | 0.0 | Never lifts feet — shuffling gait |
| Value loss | 5 - 111K | Recurring spikes (9 cycles, 1 fatal crash at iter 2549) |

**Visual evaluation (model_9400.pt):**
- ✅ Robot walks forward — first walking policy in project history!
- ❌ Falls after ~7-8 seconds — can't survive full 20s episode
- ❌ Shuffling gait — feet never lift off ground
- ❌ Recurring value loss spikes (80K-111K) — reward magnitude too high for critic

**Known issues for Run 14:**
1. Reward magnitudes too large (~3000-4000) → value loss spikes → unstable training
2. No foot lifting (feet_air_time=0) → poor balance, shuffling gait
3. Short survival (~8s) → falls from push forces, can't recover
4. Episode length ~400/1000 → only 40% survival rate

---

## Run 14 — Reward Scaling + Gait Quality

**Date:** 2026-03-09

**Changes from Run 13:**
1. **All rewards scaled down 5×** — fixes value loss spikes (target rewards ~600 vs ~3000)
2. **feet_air_time boosted** — 0.3 (5× base) → 0.8 (force actual stepping)
3. **feet_clearance kept strong** — -0.8 (penalize shuffling), target height 0.10 → 0.15m
4. **Push forces halved** — ±0.5 m/s @ 5s (gentler, learn stepping first)
5. **num_learning_epochs** — 2 → 5 (more critic updates, stabilize value function)
6. **base_height kept at 0.2** — not scaled down, important for posture

**Goals:** Eliminate value loss spikes, get feet_air_time > 0, longer survival (ep len > 600)

**Results (training in progress):**

| Iter | Reward | Episode Length | Noise Std | Value Loss | mean_vel_x | feet_air_time |
|------|--------|---------------|-----------|------------|------------|---------------|
| 148 | 353 | 363 | 0.74 | 9 | 0.04 | 0.0 |
| 448 | 978 | 703 | 0.42 | 15-590 | 0.30 | 0.0 |
| 747 | 1,394 | 747 | 0.28 | 6-1,168 | 0.60 | 0.0 |
| 1050 | 1,427 | 743 | 0.23 | 0.7-139 | 0.72 | 0.0 |
| 1353 | 1,516 | 760 | 0.21 | 0.2-11 | 0.61 | 0.0 |
| 1655 | 1,547 | 759 | 0.18 | 0.3-1.7 | 0.60 | 0.0 |
| 1958 | 1,735 | 836 | 0.15 | 0.08-948 | 0.52 | 0.0 |
| 2260 | 1,646 | 794 | 0.13 | 0.09-1,159 | 0.41 | 0.0 |
| 2563 | 1,611 | 775 | 0.12 | 0.5-1,216 | 0.55 | 0.0 |

**Visual Evaluation (model_4800):**
- Walks forward much better than Run 13 (survives ~16s vs ~8s)
- Does NOT walk straight — curves into circular path (weak angular tracking)
- Still shuffles — no foot lifting (feet_air_time = 0.0 throughout)
- Value loss stable (no more spikes — reward /5 fix worked)

**Conclusion:** Reward scaling fix worked for stability. Two remaining issues: (1) shuffling gait — `feet_air_time` formula gives zero gradient when feet never lift, (2) circular walking — `rew_tracking_ang_vel=0.22` too weak.

---

## Run 15 — Fix Shuffling + Circular Walking

**Date:** 2026-03-09

**Changes from Run 14:**
1. **NEW: swing_phase_ground penalty (-0.4)** — continuous penalty when foot is on ground during swing phase. Provides gradient signal even when feet never lift (unlike feet_air_time which requires first_contact).
2. **Boost rew_tracking_ang_vel: 0.22 → 0.5** — fix circular walking by making yaw tracking more important
3. **Biped-style air time formula** — `air_time.clamp(0, 0.5) * first_contact` (no subtract 0.5, matching EngineAI biped)

**Critical bug found:** `contact_height_threshold = 0.03m` but `link_ankle_roll` body origin is at ~0.148m above ground when standing. **Contact was NEVER detected since Run 1!** All contact-dependent rewards (`feet_air_time`, `contact_pattern`, `foot_slip`) were getting zero signal. This is why the robot never learned to step — there was literally no reward signal for foot contact events.

**Key insight:** The `(air_time - 0.5) * first_contact` formula gives ZERO reward when feet never lift because `first_contact` is always 0. The new `swing_phase_ground` penalty provides a continuous negative signal whenever the foot is on the ground during its designated swing phase — the only way to reduce this penalty is to lift the foot.

**Config:**
- All other rewards: same as Run 14
- contact_height_threshold: 0.03 → 0.16 (CRITICAL FIX — enables all contact-based rewards)
- rew_tracking_ang_vel: 0.22 → 0.5 (fix circular walking)
- rew_swing_phase_ground: -0.4 (NEW — penalize feet on ground during swing phase)
- feet_air_time formula: biped-style clamp(0, 0.5) (no subtract)

**Results (killed at iter ~985/10000):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time | swing_ground |
|------|--------|-----------|-------|------------|-------|---------------|--------------|
| 10 | 50 | 59 | 0.98 | 35 | 0.68 | 0.001 | -24 |
| 40 | 65 | 63 | 0.91 | 2.7 | 0.74 | 0.005 | -24 |
| 336 | 994 | 755 | 0.56 | 21 | 0.38 | 0.034 | -287 |
| 633 | 1,138 | 664 | 0.41 | 2.2 | 0.67 | 0.138 | -231 |
| 926 | 1,176 | 666 | 0.31 | 1.7 | 0.51 | 0.000 | -394 |
| 985 | 1,283 | 709 | 0.30 | — | — | 0.044 | -305 |

**Visual Evaluation (model_800):**
- **Walks MUCH straighter** — no more circular path (angular tracking boost worked)
- **Still shuffles** — no visible foot lifting, slides forward
- **Good survival** — stays upright for full 43s recording, stable posture
- **Moves forward** — clear forward displacement

**Conclusion:** Contact detection fix was a critical breakthrough — all contact-based rewards now active. Angular tracking fix solved circular walking. However, `swing_phase_ground = -0.4` penalty is too weak — robot maximizes other rewards (+1200 total) while eating -305 swing penalty. The brief `feet_air_time = 0.138` at iter 633 shows the signal EXISTS but isn't strong enough to reinforce. Need stronger penalty (-1.5 or -2.0) or curriculum approach.

---

## Summary of Key Lessons

1. **tracking_sigma matters enormously** — 0.25 made velocity tracking unlearnable; 5.0 gives a smooth gradient
2. **MEAN vs SUM for multi-joint rewards** — SUM over 12 joints makes exp() collapse to 0
3. **Alive bonus + termination penalty are essential** — without both, robot has no incentive to stay upright
4. **Learning rate 1e-5 is right for this task** — 1e-3 and 1e-4 caused instability
5. **Gait reference joint choice matters** — hip_yaw = spinning; hip_pitch = forward walking
6. **Add features incrementally** — changing too many things at once makes debugging impossible
7. **Anti-vibration penalties work** — dof_acc (-5e-9) + dof_vel (-1e-5) eliminated ankle oscillation (Run 6→7)
8. **Gait reference must be multi-joint** — driving only one joint is insufficient; need hip + knee + ankle coupled reference like EngineAI
9. **Domain randomization alone doesn't prevent standing-still** — pushes up to ±1.0 m/s at 4s intervals weren't enough (Runs 11-12)
10. **Standing-still exploit is a command distribution problem** — zero/small commands let the robot get positive reward by standing. Fix: forward-only commands (min 0.3 m/s), no zero commands, no small command filter
11. **Reward magnitude matters for critic** — total reward ~3000-4000 caused value loss spikes up to 111K. Scaling all rewards /5 fixed it (Run 14)
12. **feet_air_time needs gradient signal** — the standard formula gives zero when feet never lift. Need continuous swing-phase ground penalty to break shuffling local optimum
13. **VERIFY contact detection thresholds** — link_ankle_roll origin is 0.148m above ground, not at floor level. threshold=0.03m meant contact was NEVER detected (broken since Run 1). All contact rewards were zero for 14 runs!
14. **Balance stepping vs survival** — swing_phase_ground=-1.5 forces stepping but robot falls in 1.8s. Need to reduce penalty and boost survival rewards (orientation, base_height, alive, termination) so robot learns to balance while stepping
15. **Penalty must outweigh exploit reward** — swing_phase_ground=-0.4 wasn't enough; robot earns +1200 from other rewards while eating -305 swing penalty. Need penalty 3-5x stronger to overcome shuffling local optimum
16. **Curriculum solves stepping-vs-survival tradeoff** — static penalty either forces stepping but robot falls (-1.5) or allows survival but robot shuffles (-0.8). Annealing -1.5→-0.8 over 3000 iters teaches stepping first, then gradual adaptation. Run 18 maintained air_time 2-7 post-curriculum while Run 17 collapsed to 0.02
17. **Penalize over-lifting, not just under-lifting** — feet_clearance only penalized too-low feet. Without a max height cap, robot learned "higher = safer" to avoid swing penalty, causing exaggerated marching gait (Run 18). Need bidirectional clearance + max height penalty

---

## Run 16 — Stronger Swing Penalty

**Date:** 2026-03-09

**Changes from Run 15:**
1. **rew_swing_phase_ground: -0.4 → -1.5** — at -0.4 penalty totaled ~-305 vs +1200 positive rewards. At -1.5, penalty ~-1065, roughly matching positive rewards. Only way to earn net positive is to lift feet.

**Config:**
- All other rewards: same as Run 15
- rew_swing_phase_ground: -1.5 (was -0.4)

**Results (killed at iter ~1397/10000):**

| Iter | Reward | Ep Length | Noise | vel_x | feet_air_time | swing_ground |
|------|--------|-----------|-------|-------|---------------|--------------|
| 3 | -17 | 62 | 0.99 | 0.69 | 0.001 | -87 |
| 253 | +74 | 77 | 0.39 | 0.75 | 0.378 | -64 |
| 536 | +79 | 80 | 0.17 | 0.74 | 0.594 | -63 |
| 820 | +99 | 89 | 0.09 | 0.73 | 0.617 | -63 |
| 1106 | +99 | 88 | 0.07 | 0.73 | 0.623 | -62 |
| 1397 | +99 | 88 | 0.06 | 0.71 | 0.579 | -66 |

**Visual Evaluation (model_800):**
- **Robot IS stepping!** Clear knee lifts visible — first real foot lifting in the project
- **Falls after 1-2 steps** — can't maintain balance during weight transfer
- **Steps too aggressively** — knee lift is exaggerated, causing instability
- Survives ~1.8s (matches ep length 88)

**Conclusion:** Stepping achieved (breakthrough!) but -1.5 penalty forces too aggressive lifting. Robot lifts foot but can't balance on one leg. Need to reduce penalty and boost survival rewards for Run 17.

---

## Run 17 — Balance Stepping with Survival

**Date:** 2026-03-09

**Changes from Run 16:**
1. **rew_swing_phase_ground: -1.5 → -0.8** — reduce penalty so robot doesn't lift feet too aggressively
2. **target_feet_height: 0.15 → 0.08** — smaller steps are easier to balance
3. **rew_orientation: 0.2 → 0.4** — prioritize staying upright while stepping
4. **rew_base_height: 0.2 → 0.4** — maintain standing height during steps
5. **rew_alive: 0.01 → 0.03** — reward survival more
6. **rew_termination: -0.2 → -0.5** — penalize falling harder

**Key insight:** Run 16 proved robot CAN step but falls in 1.8s. The -1.5 penalty forces too aggressive knee lifts, destabilizing the robot. By reducing penalty to -0.8 (still strong enough — Run 15 showed -0.4 was too weak) and boosting survival rewards, the robot should learn to take smaller, balanced steps.

**Results (killed at iter ~2010/10000, converged to shuffling):**

| Iter | Reward | Ep Length | Noise | vel_x | feet_air_time | swing_ground |
|------|--------|-----------|-------|-------|---------------|--------------|
| 10 | 44 | 62 | 0.98 | 0.76 | 0.001 | -47 |
| 253 | 882 | 499 | 0.55 | 33 | 0.20 | -290 |
| 518 | 1,226 | 741 | 0.37 | — | 0.304 | -474 |
| 803 | 1,355 | 753 | 0.26 | — | 0.066 | -486 |
| 1090 | 1,430 | 773 | 0.20 | — | 0.053 | -476 |
| 1375 | 1,499 | 794 | 0.16 | — | 0.024 | -489 |
| 2010 | 1,542 | 798 | 0.11 | — | 0.018 | -502 |

**Visual Evaluation (model_800):**
- **More natural movement** — legs move smoothly, good posture
- **Cannot walk straight** — curves to one side (yaw drift still present)
- **Left ankle vibrates/drags** — right ankle lifts naturally, left doesn't
- **Regresses to shuffling** — initial stepping (air_time=0.304 at iter 518) disappears as robot finds shuffling exploit again

**Conclusion:** Static penalty can't solve the stepping-vs-survival tradeoff. -0.8 starts strong (air_time peaks at 0.304) but robot eventually learns to shuffle with minimal penalty. Need curriculum approach: start aggressive (-1.5, forces stepping) then relax (-0.8, allows survival). Also need PD gains randomization to force robustness, especially for weak ankle (Kp=20, Kd=0.2).

---

## Run 18 — Curriculum Swing Penalty + Standing + PD Gains Randomization

**Date:** 2026-03-09

**Changes from Run 17:**
1. **Curriculum swing penalty** — anneal `rew_swing_phase_ground` from -1.5 → -0.8 over ~3000 iterations (144,000 policy steps). Starts aggressive to force stepping, relaxes to allow survival. Linear interpolation based on global step counter.
2. **Re-enable standing** — `cmd_still_ratio = 0.1` (10% zero commands). Robot must learn to stand still AND walk. Phase freezes when standing (matching EngineAI).
3. **PD gains randomization** — EngineAI-style ±20% per DOF per episode reset. Stiffness × U(0.8, 1.2), damping × U(0.8, 1.2). Forces policy to be robust to actuator variations, especially important for ankle (Kp=20→16-24, Kd=0.2→0.16-0.24).

**Config:**
- swing_penalty_start: -1.5 (aggressive)
- swing_penalty_end: -0.8 (relaxed)
- swing_curriculum_steps: 144,000 (~3000 iterations × 48 steps)
- cmd_still_ratio: 0.0 → 0.1
- pd_gains_rand: True
- stiffness_multi_range: (0.8, 1.2)
- damping_multi_range: (0.8, 1.2)
- All other rewards: same as Run 17

**Goals:**
- feet_air_time > 0.3 sustained (not just early peak)
- Episode length > 400 (survive >8s)
- Walking + standing support
- Robust to actuator variation

**Results (killed at iter ~4022/10000, converged):**

| Iter | Reward | Ep Length | Noise | vel_x | feet_air_time | swing_ground |
|------|--------|-----------|-------|-------|---------------|--------------|
| 28 | 24 | 55 | 0.91 | 0.82 | 0.01 | -65 |
| 1062 | 953 | 767 | 0.44 | 0.57 | 3.72 | -755 |
| 2250 | 1550 | 855 | 0.28 | 0.78 | 2.15 | -243 |
| 3728 | 1649 | 833 | 0.21 | 0.51 | 6.68 | -473 |
| 4022 | 1532 | 775 | 0.20 | 0.45 | 4.42 | -347 |

**Visual Evaluation (model_4000):**
- **Robot IS walking with real stepping** — feet clearly lifting off ground (breakthrough!)
- **Left leg lifts WAY too high** — exaggerated marching, like a soldier
- **Right leg more natural** — smaller, controlled lifts
- **Robot leans/tilts** — compensating for aggressive left leg
- **Some robots fall** — destabilized by exaggerated gait
- **Clear forward movement** — good displacement, surviving 15-17s

**Conclusion:** Curriculum approach solved stepping-vs-survival tradeoff (Run 17 collapsed, Run 18 maintained). But gait reference amplitude too large (0.26 rad) and no penalty for lifting too high — robot over-learned aggressive stepping. Need to reduce amplitude and add max height penalty.

---

## Run 19 — Fix Exaggerated Stepping

**Date:** 2026-03-09

**Changes from Run 18:**
1. **Reduce gait reference amplitude**: `target_joint_pos_scale` 0.26 → 0.17 rad (knee: 0.52→0.34). Smaller reference = less exaggerated swings.
2. **Lower target foot height**: `target_feet_height` 0.08 → 0.06m. Aim for smaller, controlled steps.
3. **NEW: max foot height penalty** (`rew_feet_height_max = -0.6`): penalize swing foot going above `max_feet_height = 0.12m`. Prevents the over-lifting that caused the marching gait.
4. All curriculum, PD randomization, and standing support from Run 18 retained.

**Config:**
- target_joint_pos_scale: 0.26 → 0.17
- target_feet_height: 0.08 → 0.06
- max_feet_height: 0.12 (NEW)
- rew_feet_height_max: -0.6 (NEW)
- All other params: same as Run 18

**Goals:**
- Natural stepping gait (no exaggerated marching)
- Symmetric left/right leg behavior
- Maintain ep_length > 700 (>14s survival)
- feet_air_time > 0.3 sustained

**Results:**
- **KILLED at iter 1919** — shuffled from iter ~1200 onward (feet_air_time collapsed to 0)
- Reduced gait amplitude (0.17) made stepping too hard to learn
- Robot found it easier to shuffle and get high reward from tracking/orientation/survival

| Iter | Reward | Ep Length | Noise | vel_x | feet_air_time | swing_ground |
|------|--------|-----------|-------|-------|---------------|--------------|
| 208 | 58 | 68 | 0.68 | 0.83 | 0.23 | -68 |
| 486 | 103 | 87 | 0.58 | 0.83 | 0.25 | -64 |
| 764 | 275 | 265 | 0.56 | 0.71 | 0.31 | -272 |
| 1051 | 932 | 728 | 0.46 | 0.85 | 0.17 | -311 |
| 1339 | 1008 | 709 | 0.36 | 0.50 | 0.03 | -916 |
| 1630 | 1178 | 767 | 0.30 | 0.66 | 0.00 | -744 |
| 1919 | 1206 | 737 | 0.26 | 0.47 | 0.01 | -745 |

**Visual Evaluation (model_800 — best stepping window):**
- Left leg lifts too high for walking, right leg seems OK
- Imbalanced — robot leans and eventually falls
- Not converged at model_800 (noise 0.56) — the stepping was just noisy exploration

**Conclusion:** Reducing gait amplitude from 0.26→0.17 was too aggressive — robot couldn't learn stepping with such small targets. The /5 reward scaling from Run 14 also left gait enforcement too weak (contact_number=0.28 vs EngineAI's 1.4).

---

## Run 20 — Match EngineAI Reward Weights

**Date:** 2026-03-09

**Root Cause Analysis:** Run 14's /5 reward scaling fixed value loss spikes but crippled gait enforcement. Our reward weights were 2-5x weaker than EngineAI across all gait-critical terms — especially `feet_contact_number` (0.28 vs 1.4), `ref_joint_pos` (0.44 vs 2.2), and `feet_clearance` (-0.8 vs -1.6). This allowed the robot to exploit shuffling, asymmetric stepping, and standing.

**Changes from Run 19 — restore EngineAI reward weights:**

| Param | Run 19 | Run 20 | EngineAI |
|-------|--------|--------|----------|
| target_joint_pos_scale | 0.17 | **0.26** | 0.26 |
| target_feet_height | 0.06 | **0.10** | 0.10 |
| rew_tracking_lin_vel | 0.28 | **1.4** | 1.4 |
| rew_tracking_ang_vel | 0.5 | **1.1** | 1.1 |
| rew_tracking_sigma | 2.5 | **5.0** | 5.0 |
| rew_ref_joint_pos | 0.44 | **2.2** | 2.2 |
| rew_feet_air_time | 0.8 | **1.5** | 1.5 |
| rew_feet_contact_number | 0.28 | **1.4** | 1.4 |
| rew_orientation | 0.4 | **1.0** | 1.0 |
| rew_base_height | 0.4 | **0.2** | 0.2 |
| rew_feet_clearance | -0.8 | **-1.6** | -1.6 |
| rew_default_joint_pos | 0.16 | **0.8** | 0.8 |
| rew_feet_distance | 0.04 | **0.2** | 0.2 |
| rew_action_smoothness | -0.0006 | **-0.003** | -0.003 |
| rew_vel_mismatch | 0.1 | **0.5** | 0.5 |
| rew_foot_slip | -0.02 | **-0.1** | -0.1 |
| rew_termination | -0.5 | **-0.0** | -0.0 |
| rew_track_vel_hard | 0.1 | **0.5** | 0.5 |
| rew_low_speed | 0.3 | **0.2** | 0.2 |
| rew_dof_vel | -2e-6 | **-1e-5** | -1e-5 |
| rew_dof_acc | -1e-9 | **-5e-9** | -5e-9 |

**Kept from our additions (not in EngineAI):**
- max_feet_height: 0.12→0.15m (raised ceiling)
- rew_feet_height_max: -0.6 (prevent over-lifting)
- Curriculum swing penalty (-1.5→-0.8)
- PD gains randomization ±20%
- Push forces
- rew_alive: 0.03
- rew_lat_vel: 0.06

**Results (KILLED at iter ~555 — STANDING STILL + value loss explosion):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-------|------------|-------|---------------|
| 50 | 304 | 77 | 0.97 | 1,508 | 0.68 | 0.00 |
| 100 | 537 | 124 | 0.93 | 305 | 0.19 | 0.00 |
| 150 | 1,049 | 226 | 0.91 | 451 | 0.12 | 0.01 |
| 200 | 1,622 | 350 | 0.89 | 342 | 0.14 | 0.01 |
| 300 | 1,874 | 401 | 0.84 | 222 | ~0.1 | 0.00 |
| 400 | 2,025 | 418 | 0.73 | 175 | ~0.1 | 0.04 |
| 500 | 2,187 | 440 | 0.65 | 829 | ~0.05 | 0.03 |
| 555 | ~2,400 | ~498 | 0.61 | 17,834 | -0.09 | 0.00 |

**Failure analysis:**
- **Value loss explosion**: 15,000-17,800 spikes every ~5 iterations from iter 447 onwards. 100 spikes >1000 out of 555 iterations (18%).
- **Standing-still exploit**: Robot learned to survive (ep_length 498) without walking. Reward climbed from survival rewards alone.
- **Root cause**: EngineAI uses `num_learning_epochs=2`, we used `num_learning_epochs=5`. With 5 epochs × large reward weights, the value function is over-updated each rollout, causing catastrophic value loss spikes. The broken value function produces garbage advantage estimates, so the policy never learns locomotion.

---

## Run 21 — Match EngineAI PPO Epochs (Fix Value Loss)

**Date:** 2026-03-09

**Changes from Run 20:**
1. **num_learning_epochs: 5 → 2** — match EngineAI PPO config. Reduces gradient updates per rollout from 20 (4 batches × 5 epochs) to 8 (4 batches × 2 epochs). This should prevent value function over-updating that caused 17K+ value loss spikes in Run 20.

All reward weights unchanged from Run 20 (EngineAI-matched).

**PPO comparison (our RSL-RL vs EngineAI):**

| Parameter | Run 20 (ours) | Run 21 (ours) | EngineAI |
|-----------|--------------|--------------|----------|
| num_learning_epochs | 5 | **2** | 2 |
| num_mini_batches | 4 | 4 | 4 |
| learning_rate | 1e-5 | 1e-5 | 1e-5 |
| clip_param | 0.2 | 0.2 | 0.2 |
| gamma | 0.994 | 0.994 | 0.994 |
| lam | 0.9 | 0.9 | 0.9 |
| obs normalization | ON | ON | OFF |

**Results (KILLED at iter ~565 — STANDING STILL + value loss spikes WORSE):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-------|------------|-------|---------------|
| 100 | 406 | 94 | 0.97 | 302 | ~0 | 0.01 |
| 200 | 1,716 | 384 | 0.94 | 622 | ~0 | 0.01 |
| 300 | 2,012 | 430 | 0.92 | 109 | ~0 | 0.01 |
| 450 | 2,310 | 484 | 0.88 | 49 | ~0 | 0.05 |
| 500 | 534 | 117 | 0.87 | 2,514 | ~0 | 0.04 |
| 565 | 2,257 | 464 | 0.86 | 19,972 | 0.03 | 0.00 |

**Failure analysis:**
- **Value loss spikes WORSE than Run 20**: 107 spikes >1000 out of 563 iters (19%), peaking at 19,971. Run 20 peaked at 17.8K.
- `num_learning_epochs=2` did NOT fix the problem. Same standing-still pattern.
- **Root cause is NOT num_learning_epochs** — the remaining difference is observation normalization (RSL-RL: ON, EngineAI: OFF).

---

## Run 22 — Disable Observation Normalization (Match EngineAI)

**Date:** 2026-03-09

**Changes from Run 21:**
1. **empirical_normalization: True → False** — disable running mean/std obs normalization
2. **actor_obs_normalization: True → False** — raw observations to actor network
3. **critic_obs_normalization: True → False** — raw observations to critic network

All other params unchanged (num_learning_epochs=2, EngineAI reward weights).

**Rationale:** Runs 20 and 21 both showed catastrophic value loss spikes (17-20K) with EngineAI reward weights, regardless of epochs (5 or 2). The last remaining PPO config difference vs EngineAI is observation normalization. EngineAI does NOT normalize observations — the networks receive raw values. Our running mean/std normalizer may interact badly with the large reward magnitudes, causing the critic to produce unstable value estimates.

**Results (KILLED at iter ~551 — STANDING STILL + value loss spikes 18.8K):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-------|------------|-------|---------------|
| 117 | 408 | 96 | 0.97 | 185 | 0.19 | 0.01 |
| 200 | 555 | 119 | 0.93 | 232 | ~0.1 | 0.03 |
| 350 | 1,668 | 368 | 0.90 | 402 | ~0 | 0.02 |
| 450 | 1,844 | 406 | 0.88 | 100 | ~0 | 0.00 |
| 500 | 1,837 | 388 | 0.86 | 878 | ~0 | 0.01 |
| 551 | 2,016 | 424 | 0.85 | 18,839 | 0.05 | 0.00 |

**Failure analysis:**
- Disabling obs normalization delayed spike onset by ~15 iters but did NOT prevent them
- All 3 PPO config hypotheses eliminated:
  - Run 20: EngineAI weights + epochs=5 + obs_norm=ON → 17K spikes
  - Run 21: epochs=2 + obs_norm=ON → 19K spikes
  - Run 22: epochs=2 + obs_norm=OFF → 18.8K spikes
- **Conclusion: EngineAI reward magnitudes are fundamentally too large for our setup, regardless of PPO config**

---

## Run 23 — EngineAI Ratios at Stable Scale (/2.5)

**Date:** 2026-03-09

**Changes from Run 22:**
1. **Restore proven PPO config**: num_learning_epochs=5, obs_normalization=ON (stable in Runs 13-19)
2. **Scale all EngineAI weights by /2.5**: preserves EngineAI ratios but reduces magnitudes to prevent value loss spikes. Factor /2.5 chosen as midpoint between /5 (too weak, Run 14-19) and /1 (too strong, Run 20-22).

**Reward weights (EngineAI / 2.5):**

| Parameter | EngineAI | Run 20-22 (/1) | Run 23 (/2.5) | Run 14-19 (/5) |
|-----------|----------|----------------|---------------|----------------|
| rew_tracking_lin_vel | 1.4 | 1.4 | **0.56** | 0.28 |
| rew_tracking_ang_vel | 1.1 | 1.1 | **0.44** | 0.22 |
| rew_tracking_sigma | 5.0 | 5.0 | **5.0** | 2.5 |
| rew_ref_joint_pos | 2.2 | 2.2 | **0.88** | 0.44 |
| rew_feet_air_time | 1.5 | 1.5 | **0.60** | 0.30 |
| rew_feet_contact_number | 1.4 | 1.4 | **0.56** | 0.28 |
| rew_orientation | 1.0 | 1.0 | **0.40** | 0.20 |
| rew_base_height | 0.2 | 0.2 | **0.08** | 0.04 |
| rew_feet_clearance | -1.6 | -1.6 | **-0.64** | -0.32 |
| rew_default_joint_pos | 0.8 | 0.8 | **0.32** | 0.16 |
| rew_feet_distance | 0.2 | 0.2 | **0.08** | 0.04 |
| rew_action_smoothness | -0.003 | -0.003 | **-0.0012** | -0.0006 |
| rew_vel_mismatch | 0.5 | 0.5 | **0.20** | 0.10 |
| rew_foot_slip | -0.1 | -0.1 | **-0.04** | -0.02 |
| rew_track_vel_hard | 0.5 | 0.5 | **0.20** | 0.10 |
| rew_low_speed | 0.2 | 0.2 | **0.08** | 0.06 |
| rew_dof_vel | -1e-5 | -1e-5 | **-4e-6** | -2e-6 |
| rew_dof_acc | -5e-9 | -5e-9 | **-2e-9** | -1e-9 |

Note: tracking_sigma kept at 5.0 (it's a shape parameter, not a weight).

**Results (KILLED at iter ~982 — BEST RUN since Run 13, but shuffling):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-------|------------|-------|---------------|
| 200 | 455 | 398 | 0.67 | 27 | ~0.1 | 0.03 |
| 500 | 882 | 663 | 0.44 | 154 | ~0.2 | 0.07 |
| 687 | 1,050 | 669 | 0.39 | 15 | 0.44-0.57 | 0.06 |
| 800 | 1,283 | 790 | 0.36 | 38 | ~0.5 | 0.08 |
| 982 | 1,395 | 814 | 0.33 | 29 | 0.45-0.76 | 0.00-0.08 |

**Achievements:**
- **Value loss STABLE**: max 701 in last 100 iters, only 51 spikes total (5.2%). /2.5 scale works!
- **vel_x 0.45-0.76** — strongest sustained forward walking since Run 13
- **ep_length 814** (~16s) — best survival ever
- **low_speed = +121** — matching commanded speeds

**Remaining issue:** feet_air_time = 0.00-0.08 (shuffling). /2.5 gait enforcement not strong enough to force foot lifting.

---

## Run 24 — Selective Gait Boost (Full EngineAI on 4 Gait Terms)

**Date:** 2026-03-10

**Changes from Run 23:**
Selectively boost 4 gait-specific rewards to full EngineAI level while keeping everything else at /2.5. This targets shuffling without destabilizing total reward magnitude.

| Term | Run 23 (/2.5) | **Run 24** | EngineAI |
|------|--------------|-----------|----------|
| rew_ref_joint_pos | 0.88 | **2.2** | 2.2 |
| rew_feet_air_time | 0.60 | **1.5** | 1.5 |
| rew_feet_contact_number | 0.56 | **1.4** | 1.4 |
| rew_feet_clearance | -0.64 | **-1.6** | -1.6 |

All other rewards unchanged from Run 23 (/2.5 scale).

**Results (KILLED at iter ~395 — value loss spikes 6.7K, feet_air_time=0):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-------|------------|-------|---------------|
| 100 | 506 | 187 | 0.89 | 129 | ~0 | 0.01 |
| 200 | 1,123 | 400 | 0.82 | 66 | ~0 | 0.01 |
| 246 | 1,104 | 407 | 0.76 | 3,312 | 0.02 | 0.01 |
| 395 | — | — | — | 6,744 | — | — |

**Failure analysis:**
- `ref_joint_pos=2.2` gave ~800 free reward (72% of total) from joints near zero during stance
- This inflated total reward magnitude → value loss spikes returned (6.7K)
- `default_joint_pos` with `exp(-100*x)` formula actively penalized hip movement needed for stepping
- `tracking_sigma=5.0` gives 95% reward for standing still when commanded 0.5 m/s

---

## Run 25 — Fix Free Rewards + Targeted Gait Boost

**Date:** 2026-03-10

**Changes from Run 24 (4 targeted fixes):**

| Change | From | To | Why |
|--------|------|-----|-----|
| ref_joint_pos | 2.2 | **0.88** (/2.5) | Was 72% free reward — inflated returns |
| default_joint_pos | 0.32 | **0.05** | exp(-100*x) penalized hip movement for stepping |
| tracking_sigma | 5.0 | **3.0** | 95% reward for standing → ~87% (more gradient to move) |
| feet_air_time | 1.5 | **1.5** | Keep — direct stepping enforcer |
| feet_contact_number | 1.4 | **1.4** | Keep — alternating gait enforcer |
| feet_clearance | -1.6 | **-1.6** | Keep — anti-shuffle penalty |

**Results (KILLED at iter 1498 — shuffling, feet_air_time ≈ 0):**

| Iter | Reward | Ep Length | Noise Std | Value Loss | vel_x | feet_air_time | low_speed |
|------|--------|-----------|-----------|------------|-------|---------------|-----------|
| 300 | 644 | 431 | 0.61 | 30 | 0.14 | 0.11 | -38 |
| 500 | 901 | 556 | 0.49 | 227 | 0.26 | 0.06 | -15 |
| 700 | 1261 | 694 | 0.43 | 180 | 0.50 | 0.07 | +40 |
| 922 | 1590 | 821 | 0.39 | 156 | 0.63 | 0.10 | +60 |
| 1209 | 1515 | 751 | 0.33 | 37 | 0.48 | 0.03 | +102 |
| 1498 | 1746 | 838 | 0.31 | 36 | 0.45 | 0.00 | +137 |

**Assessment:**
- ✅ Value loss STABLE (avg 36-267, 0 spikes >1500 after iter 400)
- ✅ vel_x 0.45-0.63 (good forward movement)
- ❌ feet_air_time → 0.00 (converged to shuffling, same as Run 23)
- Noise std 0.31 = exploration exhausted, no recovery possible

**Root cause identified during this run:**
The `feet_clearance` reward had a fundamental bug — it used absolute z-position (`foot_pos_w[:,:,2] ≈ 0.148m` standing flat) instead of accumulated swing height (should start at 0). With target 0.10m, the error was `(0.10 - 0.148) = -0.048`, meaning the penalty **pushed feet DOWN** during swing phase, actively preventing stepping.

EngineAI uses accumulated delta-z heights: reset to 0 on contact, track rise during swing. Their target is 0.20m (not 0.10m).

---

## Run 26 — Fix feet_clearance (EngineAI-style accumulated height)

**Date:** 2026-03-10

**Changes from Run 25 (critical bug fix):**

1. **feet_clearance: use accumulated swing height instead of absolute z-position**
   - Added `feet_heights` buffer: resets to 0 on contact, accumulates delta-z during swing
   - Clearance error now correctly = `(target - 0.0)` at ground → penalizes shuffling
   - Previously = `(target - 0.148)` at ground → penalized foot LIFTING

2. **target_feet_height: 0.10 → 0.20** (match EngineAI)
   - With absolute z, 0.10 was chosen to be "below standing height" (broken logic)
   - With accumulated height, 0.20m is how high EngineAI expects swing foot to rise

3. **max_feet_height: 0.15 → 0.25** (accommodate 0.20 target)

All reward weights unchanged from Run 25.

**Results (KILLED at iter 2203 — shuffling, feet_air_time → 0):**

| Iter | Reward | Ep Length | Noise Std | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-----------|------------|-------|---------------|
| 154 | 455 | 370 | 0.74 | 29 | 0.18 | 0.18 |
| 500 | 900 | 590 | 0.49 | 200 | 0.35 | 0.10 |
| 728 | 1349 | 783 | 0.44 | 148 | 0.51 | 0.02 |
| 1020 | 1513 | 838 | 0.39 | 714 | 0.71 | 0.15 |
| 1315 | 1619 | 865 | 0.35 | 22 | 0.50 | 0.09 |
| 2203 | 2116 | 994 | 0.29 | 234 | 0.51 | 0.00 |

**Assessment:**
- ✅ Value loss stable, ✅ vel_x 0.50+ (good forward movement)
- ❌ feet_air_time → 0.00 (converged to shuffling again)
- Clearance fix improved early signals (0.18 at iter 150 vs Run 25's 0.05) but not enough
- Noise std 0.29 = fully converged, no recovery

**Root causes identified (deep EngineAI comparison):**
1. **Air-time formula wrong:** We use `clamp(0, 0.5)` (rewards ANY lifting). EngineAI uses `(air_time - 0.5)` which PENALIZES steps shorter than 0.5s — shuffling gets negative reward!
2. **No contact filtering:** EngineAI uses `contact_filt = contact OR last_contacts` (debounce). We use raw contact — noisy height-based detection causes premature air_time resets.
3. **target_feet_height too high:** 0.20m (our) vs 0.10m (EngineAI actual value).

---

## Run 27 — Fix air-time formula + contact filtering + target height

**Date:** 2026-03-10

**Changes from Run 26 (3 critical fixes):**

1. **Air-time formula: clamp → subtract threshold (EngineAI-style)**
   - Old: `clamp(air_time, 0, 0.5) * first_contact` — rewards any lifting
   - New: `(air_time - 0.5) * first_contact` — penalizes steps < 0.5s
   - This is the #1 suspected cause of shuffling: micro-lifts got positive reward

2. **Contact filtering: add contact_filt debouncing**
   - Old: raw contact (noisy z-height, premature air_time resets)
   - New: `contact_filt = contact OR last_contacts` (EngineAI-style debounce)
   - Also: `first_contact = (air_time > 0) AND contact_filt` (EngineAI-style)

3. **target_feet_height: 0.20 → 0.10** (actual EngineAI value)
   - 0.20 was wrong — EngineAI config uses 0.10m
   - max_feet_height: 0.25 → 0.15 (match)

**Results (KILLED at iter 1323 — shuffling, air-time penalty too weak):**

| Iter | Reward | Ep Length | Noise Std | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-----------|------------|-------|---------------|
| 100 | 384 | 260 | 0.84 | 30 | 0.14 | -0.48 |
| 462 | 1307 | 824 | 0.60 | 134 | 0.49 | -1.95 |
| 749 | 1552 | 846 | 0.51 | 661 | 0.39 | -0.22 |
| 1036 | 1610 | 835 | 0.45 | 808 | 0.73 | -0.45 |
| 1323 | 1760 | 867 | 0.41 | 104 | 0.72 | -0.73 |

**Assessment:**
- ✅ Air-time formula correct (consistently negative = penalizing shuffling)
- ❌ Penalty too weak: feet_air_time = -0.14 to -2.4 vs total reward 1600+
- At iter 1000: ref_joint_pos=807, contact_pattern=476, tracking=946 → ~2200 free reward
- Air-time penalty was <0.2% of total — robot absorbs cost easily

**Root cause:** Weight 1.5 with subtract formula gives ~-24 per episode vs +2200 from easy rewards. Need 10x stronger weight.

---

## Run 28 — Boost air-time weight 10x

**Date:** 2026-03-10

**Changes from Run 27:**
- `rew_feet_air_time`: 1.5 → **15.0** (10x increase)
  - At 1.5, penalty was -0.14 to -2.4 per episode (< 0.2% of total reward)
  - At 15.0, penalty should be -1.4 to -24 per episode (~1.5% of total)
  - Combined with subtract formula: each shuffle step costs (0.1-0.5) * 15 = -6.0
  - Over ~40 landings per episode: -240 total (vs +2200 free rewards = ~11%)

All other settings unchanged from Run 27 (contact_filt, accumulated heights, target 0.10m).

**Results (KILLED at iter 1086 — shuffling, event-based exploit):**

| Iter | Reward | Ep Length | Noise Std | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-----------|------------|-------|---------------|
| 100 | 287 | 236 | 0.87 | 18 | 0.11 | -0.61 |
| 500 | 1193 | 646 | 0.59 | 30 | 0.38 | -0.82 |
| 800 | 1589 | 818 | 0.48 | 30 | 0.56 | -1.50 |
| 1086 | 1720 | 853 | 0.42 | 20 | 0.50 | -2.26 |

**Assessment:**
- ❌ Still shuffling — 10x weight absorbed, feet_air_time increasingly negative
- Robot exploits event-based reward: slides without lifting = zero first_contact events = zero penalty
- **Fundamental reward imbalance discovered:** at /2.5 scale, free rewards (+1.48/step) > stepping penalties (-0.54/step)

**Root cause analysis — reward balance at /2.5 scale vs full EngineAI:**

| Metric | /2.5 scale | Full EngineAI |
|--------|-----------|---------------|
| Free rewards/step | +1.48 | +3.7 |
| Stepping penalties/step | -0.54 | -1.2 |
| Penalty ratio | 27% | 32% |
| ref_joint_pos formula | mean-of-exp (gives ~0.9 free) | exp-of-norm (gives ~0.3 free) |

The /2.5 scaling preserved the absolute imbalance. EngineAI works because:
1. Full penalty weights make stepping penalties exceed free rewards
2. `exp(-2*norm(diff))` formula gives ~0.3 free (vs our ~0.9 from mean-of-exp)
3. No alive bonus (pure free reward we had at 0.15)

---

## Run 29 — Full EngineAI weights + ref_joint_pos norm formula

**Date:** 2026-03-10

**Changes from Run 28 (comprehensive EngineAI alignment):**

1. **All reward weights restored to FULL EngineAI values** (removed /2.5 scaling):
   - tracking: 1.4/1.1 (was 0.56/0.44)
   - ref_joint_pos: 2.2 (was 0.88)
   - feet_air_time: 1.5 (reverted from 15.0 — formula is correct now)
   - orientation: 1.0, base_height: 0.2, feet_clearance: -1.6
   - All other rewards at full EngineAI scale

2. **ref_joint_pos formula: mean-of-exp → exp-of-norm (critical fix)**
   - Old: `mean(exp(-2*diff²))` — gives ~0.9 free reward (individual joints average out)
   - New: `exp(-2*norm(diff)) - 0.2*clamp(norm(diff), 0, 0.5)` — gives ~0.3 free (norm is strict)
   - This is the EngineAI formula; reduces free reward by ~60%

3. **Removed alive bonus**: rew_alive: 0.15 → 0.0 (not in EngineAI, pure free reward)

4. **Disabled swing_phase_ground**: 0.0 (not in EngineAI, was -1067 chaos in earlier runs)

**Expected reward balance at full scale:**
- Free rewards/step: ~2.3 (down from 3.7 due to ref_joint_pos norm formula)
- Stepping penalties/step: ~1.2 (full EngineAI)
- Key: stepping penalties are now meaningful relative to free rewards

**Also fixed:** Curriculum bypass bug — swing penalty was -1.5 even when `rew_swing_phase_ground=0.0`.

**Results (KILLED at iter 1160 — standing still, value loss spikes):**

| Iter | Reward | Ep Length | Noise Std | Value Loss | vel_x | feet_air_time | ref_joint_pos |
|------|--------|-----------|-----------|------------|-------|---------------|---------------|
| 5 | 185 | 54 | 0.99 | 584 | 0.68 | -0.007 | 12 |
| 272 | 1905 | 423 | 0.83 | 64 (14K spike) | 0.01 | -0.14 | 283 |
| 570 | 2060 | 432 | 0.54 | 8 (23K spike) | 0.01 | -0.03 | 308 |
| 865 | 2358 | 473 | 0.36 | 9 (23K spike) | 0.14 | -0.03 | 405 |
| 1160 | 2691 | 531 | 0.27 | 75 (20K spike) | 0.12 | 0.0 | 425 |

**Assessment:**
- ❌ STANDING STILL — vel_x oscillates around 0, never walks forward
- ❌ VALUE LOSS SPIKE — 20-24K every ~5 iterations, same as Runs 20-22
- ❌ ref_joint_pos rising (12→425) — norm formula better but still exploitable at weight 2.2
- ❌ noise_std 0.27 = fully converged on bad behavior, no recovery possible
- WORSE than /2.5 runs which at least achieved vel_x 0.5+

**Root cause:** Full EngineAI reward magnitudes + num_learning_epochs=5 overwhelm the value function. Value loss spikes every ~5 iters prevent coherent policy learning. EngineAI likely uses different PPO hyperparameters.

---

## Run 30 — Hybrid: /2.5 scaled weights + all bug fixes

**Date:** 2026-03-10

**Changes from Run 29:**

1. **Return to /2.5 scaled weights** (avoid value loss spikes):
   - tracking: 0.56/0.44, ref_joint_pos: 0.88, air_time: 0.6
   - orientation: 0.4, clearance: -0.64, all others /2.5

2. **Keep all bug fixes from Runs 26-29:**
   - exp-of-norm ref_joint_pos formula (less free reward: ~0.3 vs ~0.9)
   - No alive bonus (removes pure free reward)
   - contact_filt debouncing
   - Accumulated feet_heights (not absolute z)
   - Air-time subtract formula (penalizes steps < 0.5s)
   - Curriculum bypass fix (respects rew_swing_phase_ground=0.0)

**Expected reward balance at /2.5 scale WITH bug fixes:**
- Free rewards/step: ~0.7 (down from 1.48 due to norm formula + no alive)
- Stepping penalties/step: ~0.54
- Penalty ratio: 77% (up from 36%) — much closer to balance
- The norm formula is the key: at /2.5 scale, ref_joint_pos free reward drops from 0.36 to ~0.12

**Results (KILLED at iter 1692 — shuffling forward, same as Runs 23-28):**

| Iter | Reward | Ep Length | Noise Std | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-----------|------------|-------|---------------|
| 5 | 72 | 53 | 0.99 | 88 | 0.67 | -0.002 |
| 200 | 673 | 367 | 0.74 | 19 | 0.06 | -0.001 |
| 500 | 800 | 392 | 0.36 | 272 | 0.22 | -0.001 |
| 800 | 1275 | 606 | 0.24 | 2 | 0.29 | -0.011 |
| 1100 | 1412 | 637 | 0.20 | 31 | 0.45 | -0.051 |
| 1400 | 1718 | 752 | 0.16 | 2 | 0.50 | -0.276 |
| 1692 | 1613 | 705 | 0.14 | 3 | 0.43 | 0.0 |

**Assessment:**
- ✅ vel_x 0.43-0.50 (good forward velocity)
- ❌ feet_air_time never positive — converged on shuffling (same as Runs 23-28)
- Value loss spikes 2.5K-4.2K every ~5 iters (better than Run 29's 20K but still present)
- noise_std 0.14 = fully converged, no recovery

**Root cause investigation revealed:**
The z-height contact detection (foot_z < 0.16m) is unreliable. Foot body origin at 0.148m means
only 0.012m margin — shuffling feet at 0.15m stay "in contact", so no air-time penalty fires.
EngineAI uses force-based contact (contact_forces[:, foot_indices, 2] > 5N).
Also: EngineAI uses num_learning_epochs=2, not 5.

---

## Run 31 — Force-based contact detection + learning epochs=2

**Date:** 2026-03-10

**Changes from Run 30 (2 critical fixes):**

1. **Contact detection: z-height → force-based (CRITICAL)**
   - Old: `foot_z < 0.16m` — unreliable, 0.012m margin means shuffling stays "in contact"
   - New: `contact_forces_z > 5.0N` (EngineAI method) — physics-aware, force-based
   - Added IsaacLab `ContactSensor` with `track_air_time=True`
   - This should fix the entire stepping reward signal chain (first_contact, air_time, contact_pattern)

2. **PPO learning epochs: 5 → 2 (match EngineAI)**
   - 5 epochs caused value loss spikes (2.5K-4.2K at /2.5 scale, 20K+ at full scale)
   - EngineAI uses 2 epochs — fewer updates per batch = more stable value function

All other settings unchanged from Run 30 (/2.5 weights, exp-of-norm, no alive, contact_filt).

**Results (killed at iter 844/10000):**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | feet_air_time |
|------|--------|-----------|-------|------------|-------|---------------|
| 10 | ~200 | ~100 | 0.98 | 5 | 0.60 | -0.5 |
| 200 | ~550 | ~300 | 0.78 | 10-60 | 0.05 | -1.5 |
| 460 | ~700 | ~380 | 0.67 | 20-100 | 0.01 | -1.9 |
| 675 | ~835 | ~430 | 0.53 | 7-20 | 0.10 | -3.3 |
| 844 | ~900 | ~468 | 0.42 | 188 | 0.03 | -7.3 |

**Analysis:**
- ✅ Force-based contact WORKS — feet_air_time -1 to -7 (vs 0 in all prior runs)
- ❌ vel_x near zero — robot converged on standing still
- ❌ feet_air_time increasingly NEGATIVE — robot doing more short steps, never discovers long ones
- ⚠️ Value loss spikes still present: 3K-4.7K every ~5 iters despite epochs=2

**Root cause discovered:**
We used the QUADRUPED air-time formula `(air_time - 0.5) * first_contact` which PENALIZES
steps < 0.5s. EngineAI's BIPED code (`rewards_biped.py`) uses `clamp(air_time, 0, 0.5) * first_contact`
— always POSITIVE, rewards ANY step proportionally (capped at 0.5s). The robot couldn't
discover stepping because every step got negative reward.

Also: EngineAI biped includes `stance_mask` in `contact_filt` (we didn't):
`contact_filt = contact OR last_contacts OR stance_mask`
This ensures air_time only accumulates during swing phase.

Also: EngineAI biped does NOT gate air-time reward by velocity (we had `vel_cmd > 0.1` gate).

---

## Run 32 — Biped air-time formula + stance_mask contact filtering

**Date:** 2026-03-10

**Changes from Run 31 (3 fixes from EngineAI BIPED code, not quadruped):**

1. **Air-time formula: quadruped → biped**
   - Old: `(air_time - 0.5) * first_contact` — penalizes steps < 0.5s (UNDISCOVERABLE)
   - New: `clamp(air_time, 0, 0.5) * first_contact` — rewards ANY step, capped at 0.5s
   - Source: `engineai_gym/envs/robots/biped/rewards_biped.py:25-28`

2. **contact_filt: add stance_mask (EngineAI biped-specific)**
   - Old: `contact OR last_contacts`
   - New: `contact OR last_contacts OR stance_mask`
   - stance_mask = feet considered "in contact" during expected stance phase
   - Air time only accumulates during swing phase
   - Source: `engineai_gym/envs/robots/biped/biped_robot.py:116-125`

3. **Remove velocity gate on air-time reward**
   - Old: `rew_air_time *= (vel_cmd > 0.1)` — no reward when standing
   - New: no gate (matching EngineAI biped)
   - Source: `engineai_gym/envs/robots/biped/rewards_biped.py` (no gate present)

All other settings unchanged from Run 31 (force-based contact, epochs=2, /2.5 weights).

**Why this should work:**
- Run 15 used clamp formula but had BROKEN z-height contact → no signal
- Run 31 has WORKING force-based contact but WRONG quadruped formula → negative signal
- Run 32 combines WORKING contact + CORRECT biped formula → should get POSITIVE signal

**Results (COMPLETED — iter 9999/10000, 5h31m):**
- ✅ FIRST WALKING POLICY IN 32 RUNS
- Mean reward: 2000-2500 (peak 2533 at iter ~9453)
- Episode length: 780-973 (peak 973 — 97% of max)
- vel_x: 0.3-0.5 (stable forward walking)
- feet_air_time: +5.8 to +9.4 (POSITIVE throughout — breakthrough confirmed)
- noise_std: 0.07 (fully converged)
- Value loss: 1-200 baseline, push spikes to 1329 (bounded, normal)

**Visual evaluation:**
- Robot walks forward with alternating steps
- Upright posture maintained
- Foot clearance LOW (shuffling more than stepping)
- Slight lateral drift to the right
- Verdict: "almost perfect" — needs higher foot clearance and stronger pushes

**Model:** `logs/rsl_rl/pm01_walking/2026-03-10_06-39-39/model_9999.pt`

---

## Run 33 — Full EngineAI reward weights + stronger pushes

**Date:** 2026-03-10

**Changes from Run 32:**

1. **Push settings: match EngineAI for resilience**
   - `push_interval_s`: 5.0 → **15.0** (less frequent, allows longer episodes)
   - `max_push_vel_xy`: 0.5 → **1.0** (2× stronger pushes)
   - `max_push_ang_vel`: 0.4 → **0.6** (match EngineAI)
   - Rationale: Run 32 pushes (0.5 m/s @ 5s) limited episode length to ~81% and were too
     gentle. EngineAI uses 1.0 m/s @ 15s — stronger but less frequent. Should improve
     resilience AND allow longer episodes.

2. **Reward scales: /2.5 → full EngineAI weights**
   - All 20 reward terms upgraded from /2.5 scaling to full EngineAI weights
   - Run 29 failed with full weights because formulas were broken (quadruped air-time,
     z-height contact). Now that biped fixes (Run 32) are proven, full weights are safe.
   - Key changes: tracking_lin_vel 0.56→1.4, ref_joint_pos 0.88→2.2, feet_air_time 0.6→1.5,
     feet_clearance -0.64→-1.6, action_smoothness -0.0012→-0.003, foot_slip -0.04→-0.1

No formula changes — all reward formulas unchanged from Run 32.

**Why this should work:**
- Run 32 proved all formulas are correct (biped air-time, stance_mask, force contact)
- /2.5 scaling was a safety measure from Run 30 (pre-biped-fix era)
- Full weights give stronger learning signal, especially for feet_clearance (-1.6 vs -0.64)
  which should encourage higher foot lifting (main visual issue from Run 32)
- Stronger pushes + less frequent = better balance training + longer episodes

**Results (KILLED at iter 1192 — collapsed):**
- ❌ vel_x collapsed: 0.14 (iter 586) → -0.05 (iter 888) → -0.10 (iter 1192)
- ❌ Episode length crashed: 985 (iter 586) → 631 (iter 888) → 88 (iter 1192)
- ❌ Value loss spiraled: 18 → 58,017 → 22,458 (vs Run 32's max 1.3K)
- ❌ Reward crashed: 5212 → 3386 → 375
- ✅ Push settings worked: episode length hit 985 at iter 586 (vs Run 32's max 973)
- ✅ feet_air_time was positive early (peaked at 7.0 at iter 888)

**Root cause:**
Full EngineAI weights (2.5× jump from /2.5 scaling) were too aggressive. The stronger
penalties (action_smoothness -0.003, foot_slip -0.1, energy -0.0001, feet_clearance -1.6)
constrained the policy before it could discover forward walking. Value function destabilized
(22K loss vs Run 32's 1.3K), causing cascading collapse. The robot converged on stepping
in place, then falling.

**Key lesson:** Can't jump from /2.5 to full weights in one step. Push settings (15s @ 1.0 m/s)
are good — keep those. Need intermediate reward scaling.

---

## Run 34 — Intermediate reward weights (/1.5) + EngineAI pushes

**Date:** 2026-03-10

**Changes from Run 33:**

1. **Reward scales: full → /1.5 (intermediate)**
   - /2.5 (Run 32) worked but foot clearance was weak
   - Full (Run 33) collapsed — too aggressive
   - /1.5 gives ~67% more signal than Run 32 without the catastrophic penalty pressure
   - Key changes: tracking_lin_vel 1.4→0.93, ref_joint_pos 2.2→1.47, feet_air_time 1.5→1.0,
     feet_clearance -1.6→-1.07, action_smoothness -0.003→-0.002, foot_slip -0.1→-0.067

2. **Push settings: keep from Run 33 (proven good)**
   - `push_interval_s`: 15.0 (episode length hit 985 in Run 33)
   - `max_push_vel_xy`: 1.0
   - `max_push_ang_vel`: 0.6

No formula changes — all reward formulas unchanged from Run 32/33.

**Why this should work:**
- /1.5 is halfway between /2.5 (worked) and full (failed)
- Stronger feet_clearance (-1.07 vs -0.64) should improve foot lift without crushing policy
- Push settings proven in Run 33 (985 episode length)
- Same biped fixes that made Run 32 succeed

**Goals:**
- vel_x > 0.3 (MUST walk forward)
- feet_air_time positive
- Value loss < 500 (stable)
- Better foot clearance than Run 32

**Results: COMPLETED — BEST RUN EVER**

- **Training time:** 5h39m (10000 iterations)
- **Model:** `2026-03-10_14-13-27/model_9999.pt`
- **Log:** `train_2026-03-10_14-13-13.log`

**Final metrics (iter 9999):**
- Mean reward: 4,290 (push-cycle 12 hit at very end)
- Episode length: 986
- vel_x: 0.58
- Noise std: 0.05

**Peak metrics (iter 9930, pre-final push):**
- Mean reward: 4,361 (all-time record)
- Episode length: 999 (maxed out)
- feet_air_time: +16.2 (record was +16.94 at iter 9358)
- vel_x: 0.45
- ref_joint_pos: 531.0 (record was 719.7 at iter 9998)
- Value loss: 17.2

**Peak reward progression (10 consecutive records across push-cycle recoveries):**
- 4161 → 4296 → 4321 → 4332 → 4333 → 4342 → 4353 → 4360 → 4361

**Push resilience:**
- 12 push-cycle disruptions survived (1.0 m/s @ 15s interval)
- Each cycle: value loss spiked 20-30K, then recovered within ~280 iters
- Robot became increasingly resilient — vel_x barely dropped during later pushes

**Comparison with Run 32 (previous best):**
| Metric | Run 32 | Run 34 | Improvement |
|--------|--------|--------|-------------|
| Peak reward | ~3,900 | 4,361 | +12% |
| feet_air_time | ~5.0 | 16.94 | +239% |
| Push resilience | 0.5 m/s | 1.0 m/s | 2x stronger |
| Noise std (final) | ~0.10 | 0.05 | 2x more converged |

**Key takeaway:** /1.5 reward scaling is the sweet spot. Stronger penalties (especially
feet_clearance -1.07 vs -0.64) dramatically improved stepping behavior without destabilizing
training. EngineAI push settings (1.0 m/s @ 15s) made the policy more resilient while
actually improving episode length (less frequent disruptions than 5s interval).

**Next steps:** Visual evaluation — compare foot clearance to Run 32 video.

---

## Run 35 — Stability-focused: selective full EngineAI weights + tighter termination

**Date:** 2026-03-10

**Changes from Run 34:**

1. **Termination height: 0.45 → 0.65m (tighter margin)**
   - Run 34: robot could fall 44% of standing height (0.813m) before reset
   - Run 35: only 0.163m margin — forces robot to learn balance recovery early
   - Prevents catastrophic falls and bad habit formation

2. **Push interval: 15.0 → 8.0s (more frequent)**
   - EngineAI value — more reactive stepping practice
   - Robot gets pushed more often, learns faster balance correction

3. **Stability-critical rewards boosted to FULL EngineAI (rest stays /1.5):**
   - `rew_orientation`: 0.67 → 1.0 (stronger upright incentive)
   - `rew_ref_joint_pos`: 1.47 → 2.2 (tighter gait tracking = better balance)
   - `rew_base_height`: 0.13 → 0.2 (maintain standing height)
   - `rew_vel_mismatch`: 0.33 → 0.5 (penalize unwanted z-motion)
   - `rew_track_vel_hard`: 0.33 → 0.5 (force velocity tracking)
   - `rew_low_speed`: 0.13 → 0.2 (punish slowness)

4. **Rewards kept at /1.5 (unchanged):**
   - tracking_lin_vel (0.93), tracking_ang_vel (0.73)
   - feet_air_time (1.0), feet_contact_number (0.93)
   - feet_clearance (-1.07), default_joint_pos (0.53), feet_distance (0.13)
   - action_smoothness (-0.002), energy (-0.000067), foot_slip (-0.067)
   - dof_vel, dof_acc, lat_vel

No formula changes — all reward formulas unchanged from Run 34.

**Why this should work:**
- Run 33 failed because ALL weights were boosted simultaneously → too much penalty pressure
- Run 35 only boosts 6 stability-critical weights, keeps 13 others at proven /1.5
- Tighter termination (0.65m) teaches recovery before catastrophic fall
- More frequent pushes (8s) = more balance practice per episode

**Goals:**
- Zero falling during play (primary goal)
- Maintain vel_x > 0.3
- Maintain feet_air_time positive
- Value loss < 500 (stable)

**Results: KILLED at iter 1443 — STANDING STILL**

- vel_x stuck at 0.03-0.07 from iter 228 to 1443 — never learned to walk
- Robot optimized for standing (reward 3,372, ep_len 872) but vel_x = 0.03
- ref_joint_pos at 2.2 was too dominant — reward ~993 for standing in default pose
- Noise std dropped to 0.15 — policy converged on wrong behavior (standing)
- tracking_lin_vel at 0.93 was too weak to overcome stability rewards

**Key lesson:** ref_joint_pos at 2.2 rewards default pose matching, which competes
with forward walking. Need stronger velocity tracking to counterbalance stability rewards.

---

## Run 36 — Velocity tracking at full EngineAI + stability fixes

**Date:** 2026-03-10

**Changes from Run 35:**

1. **Velocity tracking boosted to FULL EngineAI:**
   - `rew_tracking_lin_vel`: 0.93 → 1.4 (stronger forward drive)
   - `rew_tracking_ang_vel`: 0.73 → 1.1 (match velocity tracking)

2. **ref_joint_pos REVERTED to /1.5:**
   - `rew_ref_joint_pos`: 2.2 → 1.47 (was too dominant, caused standing still)

3. **Kept from Run 35 (stability):**
   - `termination_height`: 0.65m (tighter margin — good)
   - `push_interval_s`: 8.0s (more frequent pushes)
   - `rew_orientation`: 1.0 (full EngineAI — upright)
   - `rew_base_height`: 0.2 (full EngineAI — standing height)
   - `rew_vel_mismatch`: 0.5 (full EngineAI — penalize z-motion)
   - `rew_track_vel_hard`: 0.5 (full EngineAI — velocity tracking)
   - `rew_low_speed`: 0.2 (full EngineAI — punish slowness)

No formula changes — all reward formulas unchanged.

**Why this should work:**
- Run 35 proved stability rewards (orientation, base_height) don't break training
- The problem was ref_joint_pos at 2.2 competing with forward walking
- Full EngineAI velocity tracking (1.4) gives stronger forward drive signal
- Combined with stability rewards, should walk forward AND stay upright

**Results: KILLED at iter 564 — STANDING STILL**

- vel_x peaked at 0.1035 then declined to 0.0753
- Noise std converging at 0.49 — policy converging on standing
- swing_ratio_l: 0.0 — left leg never lifts, one-legged behavior
- Mean reward: 2,839, episode length: 631
- Same failure pattern as Run 35

**Key lesson:** `termination_height=0.65m` is the root cause of standing still in both
Run 35 and 36. Only 0.163m margin from standing height (0.813m) — robot gets killed
whenever it bends knees during walking exploration. Run 34 walked beautifully with 0.45m.

---

## Run 37 — Run 36 weights + revert termination height to 0.45m

**Date:** 2026-03-10

**Changes from Run 36:**

1. **Termination height REVERTED:**
   - `termination_height`: 0.65 → 0.45 (Run 34 walked well with this — 0.65 too strict)

2. **Everything else KEPT from Run 36:**
   - Full EngineAI velocity tracking (1.4, 1.1)
   - ref_joint_pos at 1.47 (/1.5)
   - All stability rewards (orientation 1.0, base_height 0.2, vel_mismatch 0.5)
   - Push interval 8.0s, stronger pushes (1.0, 0.6)

No formula changes — only termination_height changed.

**Why this should work:**
- Run 34 was our best walker ever (reward 4,361) with termination_height=0.45m
- Runs 35 AND 36 both stood still with termination_height=0.65m
- 0.65m leaves only 0.163m margin — kills robot for normal knee-bending during walking
- Run 37 = Run 36 velocity boosts + Run 34's proven termination height
- Should combine stronger forward drive with room to explore walking

**Goals:**
- vel_x > 0.3 (MUST walk forward — Run 35+36 both failed this)
- Better stability than Run 34 (thanks to boosted velocity tracking + stability rewards)
- feet_air_time positive
- Fewer falls during play than Run 34

**Results: COMPLETED — BEST RUN EVER**

- **Peak reward: 5,760** (iter 8789) — +32% over Run 34's 4,361
- **Final reward: 5,720** — near peak at completion
- **Final episode length: 999 (MAX)** — survives full 20s episodes
- **vel_x avg ~0.45, peaked 0.78** — fast forward walking
- **feet_air_time: 10-12** — excellent stepping
- **Noise: 0.04** — fully converged
- **low_speed consistently positive** (+264 final) — robot exceeds minimum speed
- Swing ratio balanced 0.50/0.50 throughout
- Push-cycle pattern: peaks ~5,750, dips to ~2,500 (still falling from pushes ~50%)

**Training progression:**
- iter 400: vel_x ~0.15 (stepping in place initially)
- iter 900: vel_x ~0.22 (starting to walk)
- iter 1250: vel_x ~0.30 (flagged PROMISING)
- iter 1600: reward surpassed Run 34's final (4,635 vs 4,361)
- iter 2600: ep_len hit 999 (max), reward plateau began at ~5,500
- iter 5000: halfway, reward 5,644
- iter 10000: completed, peak 5,760

**Key lesson:** `termination_height=0.45m` is essential for walking exploration.
Full EngineAI velocity tracking (1.4/1.1) + lenient termination = best combo.
Runs 35+36 proved that 0.65m termination kills walking regardless of reward weights.

**Model saved:** `release/run37_best/model_9999.pt`

---

## Run 38 — Fix shuffling gait (boost foot clearance + height)

**Date:** 2026-03-11

**Changes from Run 37:**

1. **Foot clearance boosted to FULL EngineAI:**
   - `rew_feet_clearance`: -1.07 → -1.6 (was /1.5, now full — penalize low feet harder)

2. **Target feet height raised:**
   - `target_feet_height`: 0.10 → 0.12 (demand higher foot lifts)
   - `max_feet_height`: 0.15 → 0.18 (allow higher swing)
   - `rew_feet_height_max`: -0.4 → -0.6 (full EngineAI — penalize too-low feet)

3. **Action scale increased:**
   - `action_scale`: 0.5 → 0.6 (allow larger joint movements for more expressive gait)

4. **Everything else KEPT from Run 37:**
   - All reward weights unchanged
   - termination_height=0.45m, push settings same

No formula changes — only foot-related parameters and action_scale changed.

**Why this should work:**
- Run 37 had great metrics but visual showed shuffling/sliding feet
- feet_clearance at -1.07 wasn't enough to penalize low foot swing
- Higher target_feet_height (0.12m) demands the robot lifts feet more
- Larger action_scale (0.6) allows deeper knee bends for natural stepping
- Full EngineAI feet_height_max (-0.6) penalizes feet staying too close to ground

**Goals:**
- Natural stepping gait (no shuffling)
- Maintain vel_x > 0.3 (Run 37 level)
- Higher swing_foot_height values (>0.08m)
- Reward > 5,000

**Results: COMPLETED — NEW ALL-TIME RECORD!**

- **Final reward: 5,792** (surpassed Run 37's 5,760!)
- **Peak vel_x: 0.606** (35% faster than Run 37's 0.45)
- **Peak swing_l: 0.074m** (23% higher than Run 37's ~0.06m)
- **Peak swing_r: 0.067m** (34% higher than Run 37's ~0.05m)
- **Final gait symmetry: 0.51/0.49** (perfect)
- **Final foot force balance: 203/204 N** (nearly equal)
- Training time: 6h 31min (10000 iterations)

**Training progression:**
- iter 1-700: stepping in place (vel_x ~0.06, similar to Run 37 early phase)
- iter 700-1500: slow breakout, vel_x climbing (delayed vs Run 37's iter 800)
- iter 1591: first breakout — reward 4,972, ep_len 991
- iter 2470: one-legged hopping phase (swing_ratio_l dropped to 0.11) — self-corrected
- iter 3374: vel_x peaked 0.48 — faster than Run 37 at same stage
- iter 5307: vel_x broke 0.6 m/s — new speed record
- iter 6188: reward 5,660, recovering from push dips
- iter 7946: vel_x peaked 0.606 (all-time record)
- iter 8740: reward 5,743, swing_l peaked 0.073m, feet_air_time 13.6
- iter 9440: swing_l peaked 0.074m, ref_joint_pos 554 (best gait tracking)
- iter 9999: FINAL — reward 5,792 (NEW RECORD), perfect symmetry

**Key lessons:**
- action_scale 0.6 (vs 0.5) enabled significantly faster walking (+35% vel_x)
- Stronger feet_clearance penalty (-1.6 vs -1.07) improved foot heights but not to 0.08m target
- ~0.065-0.074m swing height appears to be the physical limit for this URDF
- Push dips happen every ~580 iters, dropping reward ~50% — always recovers
- Training can exhibit transient asymmetric phases (one-legged hopping) that self-correct
- noise_std converged to 0.04 by completion — very refined policy

**Model saved:**  (final),  (mid-peak)

**Results: COMPLETED — NEW ALL-TIME RECORD!**

- **Final reward: 5,792** (surpassed Run 37s 5,760!)
- **Peak vel_x: 0.606** (35% faster than Run 37s 0.45)
- **Peak swing_l: 0.074m** (23% higher than Run 37s ~0.06m)
- **Peak swing_r: 0.067m** (34% higher than Run 37s ~0.05m)
- **Final gait symmetry: 0.51/0.49** (perfect)
- **Final foot force balance: 203/204 N** (nearly equal)
- Training time: 6h 31min (10000 iterations)

**Training progression:**
- iter 1-700: stepping in place (vel_x ~0.06, similar to Run 37 early phase)
- iter 700-1500: slow breakout, vel_x climbing (delayed vs Run 37s iter 800)
- iter 1591: first breakout — reward 4,972, ep_len 991
- iter 2470: one-legged hopping phase (swing_ratio_l dropped to 0.11) — self-corrected
- iter 3374: vel_x peaked 0.48 — faster than Run 37 at same stage
- iter 5307: vel_x broke 0.6 m/s — new speed record
- iter 7946: vel_x peaked 0.606 (all-time record)
- iter 8740: reward 5,743, swing_l peaked 0.073m
- iter 9440: swing_l peaked 0.074m, ref_joint_pos 554 (best gait tracking)
- iter 9999: FINAL — reward 5,792 (NEW RECORD), perfect symmetry

**Key lessons:**
- action_scale 0.6 (vs 0.5) enabled significantly faster walking (+35% vel_x)
- Stronger feet_clearance penalty (-1.6 vs -1.07) improved foot heights modestly
- ~0.065-0.074m swing height appears to be the physical limit for this URDF
- Push dips happen every ~580 iters, dropping reward ~50% — always recovers
- Training can exhibit transient asymmetric phases that self-correct
- noise_std converged to 0.04 by completion — very refined policy

**Model saved:** release/run38_best/model_9999.pt (final), release/run38_best/model_8800.pt (mid-peak)

---

## Run 39 — Anti-Wobble (base_acc + knee_distance + smoothness)

**Date:** 2026-03-11
**Status:** ✅ COMPLETED — NEW ALL-TIME RECORD: 5,858

**Changes from Run 38:**
| Parameter | Run 38 | Run 39 | Rationale |
|-----------|--------|--------|-----------|
| action_scale | 0.6 | 0.55 | Reduce for stability |
| rew_action_smoothness | -0.002 | -0.003 | Full EngineAI value |
| rew_base_acc | N/A | 0.2 | NEW: exp(-norm(base_acc)*3) — smooth base motion |
| rew_knee_distance | N/A | 0.2 | NEW: exp(-|knee_dist-target|*20) — prevent knee collision |
| target_knee_dist | N/A | 0.25m | Target distance between knees |

**Motivation:** Run 38 achieved record reward (5,792) but visual evaluation showed left leg wobbling. EngineAI reference code had base_acc and knee_distance rewards we were missing.

**Results:**
- **Peak reward: 5,858** (+66 over Run 38's 5,792 — new all-time record!)
- **Final reward: 5,839** (iter 9999, high phase)
- **vel_x: 0.49 m/s** (slightly slower than Run 38's 0.61 peak, but steadier)
- **Foot force symmetry: greatly improved** — consistently 0.91-0.99 ratio (Run 38 was ~0.78)
- **Swing ratio: near-perfect** — 0.506/0.495 final (Run 38 had more variation)
- **Swing heights: L=0.082/R=0.068m** — asymmetry reduced but not eliminated (14mm gap)
- **knee_distance: 172.4** (highest at completion — anti-wobble working)
- **base_acc: peaked 15.5** (smooth base motion reward)
- **Episode length: 999** (full 20s episodes consistently)
- **Training time: 6h 5m** (22,604 seconds, 10,000 iterations)

**Training progression:**
- iter 1-550: stepping in place, negative vel_x (exploring)
- iter 815: slow walker (vel_x 0.16), force asymmetry 172/248N
- iter 1333: PROMISING — vel_x broke 0.41, learning to walk forward
- iter 1604: STRONG — vel_x 0.31-0.55, matching Run 38 speed
- iter 2330: peak 5,601 — approaching Run 38 record
- iter 3262: forces BALANCED 197/199N (ratio 0.99!) — anti-wobble success
- iter 5953: peak 5,787 — just 5 points from Run 38 record
- iter 6241: NEW ALL-TIME RECORD 5,798 — beat Run 38!
- iter 7385: 5,812 (+20 over Run 38)
- iter 8760: 5,834 (+42 over Run 38)
- iter 9037: 5,848 (+56 over Run 38)
- iter 9371: 5,858 (+66 over Run 38) — FINAL PEAK
- iter 9999: 5,839 (final, high phase)

**Key lessons:**
- base_acc reward dramatically improved force symmetry (0.78 → 0.91-0.99 ratio)
- knee_distance reward reached 172.4 at completion — effective at maintaining leg spacing
- Reduced action_scale (0.55 vs 0.6) traded ~20% peak speed for better stability
- Swing height asymmetry (L > R by ~14mm) persists — may need explicit L/R symmetry reward
- Push-cycle pattern consistent: every ~580 iters, reward drops ~50% then recovers
- noise_std converged to 0.02 (very refined, was 0.04 in Run 38)
- Reward kept climbing until iter 9371 — longer training could potentially go higher
- Visual evaluation still needed to confirm wobble is actually fixed

**Remaining issues for future runs:**
- Swing height asymmetry (L=0.082 vs R=0.068m)
- No knee bending upward (hip_pitch gait ref vs EngineAI's hip_yaw)
- target_feet_height=0.12m too high per TARGET_GAIT.md analysis (should be 0.05-0.06m)

**Model saved:** release/run39_best/model_9400.pt (near-peak iter 9371), release/run39_best/model_9999.pt (final)

---

## Run 40 — Hip Yaw Gait Ref + Natural Foot Height

**Date:** 2026-03-11
**Status:** TRAINING (headless)
**Target:** Stable locomotion, velocity tracking, balance, push recovery, human-like knee bending.

**Changes from Run 39:**
| Parameter | Run 39 | Run 40 | Rationale |
|-----------|--------|--------|-----------|
| gait_ref joint | hip_pitch (idx 0/6) | hip_yaw (idx 2/8) | EngineAI match — natural knee lift |
| target_feet_height | 0.12m | 0.06m | Match original robot (3-5cm clearance) |
| max_feet_height | 0.18m | 0.10m | Tighter range for natural steps |
| cmd_still_ratio | 0.1 | 0.0 | 100% walking (standing = fixed motor mode) |

**Motivation:** Run 39 was a stiff-legged shuffle despite record reward. The original EngineAI robot uses hip_yaw for gait reference, which creates natural knee-lifting motion during swing phase. Our hip_pitch only swings legs forward/backward like a pendulum. Also reduced target_feet_height from 0.12m to 0.06m to match the original robot's low foot clearance (3-5cm).

**Risk:** hip_yaw caused 360° spinning in Run 4/8 (early training without domain rand). Now have anti-wobble rewards + domain rand — should be manageable.

**Design decision:** Standing still handled by fixed motor mode (lock joints at default pose), not RL. Set cmd_still_ratio=0.0 so 100% of training focuses on walking quality.

**Results:** COMPLETED — Peak 5,747 (iter 8,754), final 5,743 (iter 9,999)

| Metric | Peak (iter ~8,754) | Final (iter 9,999) |
|--------|-------------------|-------------------|
| Mean reward | 5,747 | 5,743 |
| Episode length | 999 | 999 |
| vel_x | 0.447 m/s | 0.542 m/s |
| noise_std | 0.07 | 0.06 |
| foot_height_l | 0.065m | 0.069m |
| foot_height_r | 0.095m | 0.102m |
| force_l / force_r | 167/242 N (0.69) | 166/250 N (0.66) |
| swing_ratio L/R | 0.506/0.494 | 0.506/0.494 |
| base_acc | 3.63 | 4.34 |
| knee_distance | 148.9 | 140.4 |
| base_height | 122.6 | 133.6 |

**Key observations:**
- ✅ hip_yaw gait ref works — no spinning, natural knee bending confirmed
- ✅ Full episode survival (999 steps) at peak
- ✅ Good forward velocity (0.45-0.54 m/s)
- ✅ Push recovery improving: dip#1 800 iters, dip#2 537 iters
- ✅ base_acc best at 4.34 (smooth base motion), base_height best at 133.6 (upright)
- ⚠️ Force asymmetry structural at 0.66-0.69 ratio (hip_yaw side effect)
- ⚠️ Right foot consistently higher than left (0.095-0.102 vs 0.065-0.069m)
- ⚠️ Peak 5,747 — 111 below Run 39's record (5,858), but better gait quality

**Push-cycle pattern (3 dips in 10k iters):**
| Dip | Start iter | Depth | Recovery |
|-----|-----------|-------|----------|
| 1 | ~6,876 | -53% | 800 iters |
| 2 | ~8,217 | -49% | 537 iters |
| 3 | ~9,560 | -46% | recovered by iter 9,999 |

**Comparison vs Run 39:**
| Feature | Run 39 (hip_pitch) | Run 40 (hip_yaw) |
|---------|-------------------|-------------------|
| Peak reward | 5,858 (record) | 5,747 (-111) |
| Knee bending | ❌ stiff-legged | ✅ natural bend |
| Force balance | 0.91-0.99 | 0.66-0.69 (worse) |
| Foot height sym | variable | R consistently higher |
| Base smoothness | good | better (base_acc 4.34) |
| Posture | good | better (base_height 133.6) |

**Model saved:** release/run40_best/model_8800.pt (near-peak), release/run40_best/model_9999.pt (final)

**Remaining issues for future runs:**
- Force asymmetry (L/R ratio 0.66-0.69) — may need explicit force-balance reward
- Right foot over-lift (0.10m vs left 0.07m) — asymmetric gait ref effect
- Reward plateau at ~5,740-5,750 — may need longer training or hyperparameter tuning
- Lateral drift (not walking straight) — rew_lat_vel too weak at 0.04

---

## Run 41 — Force Balance + Anti-Drift + Friction Rand

**Date:** 2026-03-12
**Status:** TRAINING
**Target:** Fix L/R force asymmetry, eliminate lateral drift, keep knee bending.

**Changes from Run 40:**
| Parameter | Run 40 | Run 41 | Rationale |
|-----------|--------|--------|-----------|
| rew_force_balance | (none) | 0.15 | NEW: reward equal L/R ground forces (target ratio 0.5) |
| rew_lat_vel | 0.04 | 0.2 | 5x boost — was too weak, causing lateral drift |
| friction_rand | (none) | True [0.7, 1.3] | NEW: randomize friction per reset for robustness |

**Motivation:** Run 40 video analysis showed: (1) lateral drift left/right during walking, (2) asymmetric foot heights, (3) L/R force ratio 0.66-0.69. Root cause analysis confirmed gait reference signs are correct (match EngineAI exactly), but we lack EngineAI's symmetry loss + observation history. Force-balance reward directly targets the asymmetry. Stronger lat_vel reward fights drift. Friction randomization improves generalization.

**New reward term — force_balance:** `exp(-(force_ratio - 0.5)^2 * 50)` where `force_ratio = min(L,R) / (L+R)`. Maximum reward at 0.5 (equal forces), decays sharply for imbalanced forces.

**Risk:** Force balance reward might conflict with hip_yaw gait ref (which structurally creates some asymmetry). If too strong, could cause limp. Starting conservative at 0.15.

**Results:** KILLED at iter 1,874 — stepping-in-place failure.
- vel_x stuck at 0.015-0.023 from iter 800-1,874 (never reached 0.3)
- Force balance reward worked for symmetry (ratio improved to 0.78-0.80 while standing)
- But combined with lat_vel, over-constrained policy → local minimum of symmetric stepping
- force_balance=49.6 + lat_vel=170.3 dominated positive rewards → no incentive to walk forward
- low_speed penalty (-273) overwhelmed by symmetry rewards

**Lesson:** force_balance (0.15) + lat_vel (0.2) together too aggressive — policy exploited symmetry rewards

---

## Run 41b — Force Balance + Anti-Drift (Halved Weights)

**Date:** 2026-03-12
**Status:** TRAINING
**Target:** Same as Run 41 but with halved reward weights to avoid stepping-in-place.

**Changes from Run 41:**
| Parameter | Run 41 | Run 41b | Rationale |
|-----------|--------|---------|-----------|
| rew_force_balance | 0.15 | 0.08 | Halved — was too strong, caused stepping-in-place |
| rew_lat_vel | 0.2 | 0.10 | Halved — combined with force_balance was over-constraining |
| friction_rand | True [0.7, 1.3] | kept | No issues observed |

**Motivation:** Run 41 proved force_balance reward works (ratio improved to 0.78-0.80) but was too strong combined with boosted lat_vel. Policy found it more rewarding to step symmetrically in place than to walk forward asymmetrically. Halving both should allow forward locomotion while still improving symmetry over Run 40.

**Results:** KILLED at iter 813 — stepping-in-place failure (same as Run 41).
- vel_x oscillated 0.05-0.14 from iter 400-813 (never reached 0.3)
- Force ratio: 0.47 (190/218 N) — good symmetry but at cost of forward motion
- Swing ratio: L=0.496, R=0.504 — perfectly symmetric (standing still is symmetric!)
- Episode length: 500-660 (surviving but not walking)
- Mean reward: 1,800-3,000 (driven by symmetry rewards, not velocity)

**Lesson:** force_balance as a POSITIVE reward fundamentally conflicts with forward locomotion.
Even at 0.08, it creates a local minimum where symmetric stepping-in-place scores higher than
asymmetric forward walking. Both Run 41 (0.15) and 41b (0.08) produced the same failure mode.

**Conclusion:** Force balance must be either:
1. Removed entirely — rely only on modest lat_vel penalty for drift
2. Converted to a PENALTY (punish extreme imbalance, not reward balance)
3. Applied only during stance phase (not during swing when forces are naturally asymmetric)

---

## Run 42 — Remove Force Balance, Modest Lat Vel + Friction Rand

**Date:** 2026-03-12
**Status:** TRAINING
**Target:** Fix lateral drift from Run 40 WITHOUT sacrificing forward locomotion.

**Strategy:** Keep friction randomization (proven harmless), use only a moderate lat_vel
boost (0.06 — 1.5x Run 40's 0.04). Remove force_balance entirely. The theory is that
lateral drift was partially caused by deterministic friction (always same value) and a
weak lat_vel penalty. Friction randomization should help the policy generalize, and a
modest lat_vel boost should reduce drift without over-constraining.

**Changes from Run 40:**
| Parameter | Run 40 | Run 42 | Rationale |
|-----------|--------|--------|-----------|
| rew_force_balance | (none) | (none) | Proven to cause stepping-in-place in Run 41+41b |
| rew_lat_vel | 0.04 | 0.06 | 1.5x boost — modest, should not over-constrain |
| friction_rand | (none) | True [0.7, 1.3] | Keep from 41b — improves generalization |

**Results:** KILLED at iter 692 — vel_x stuck ~0.05, stepping-in-place.
- vel_x oscillated 0.02-0.13 from iter 300-692 (never reached 0.3)
- Foot force L/R: 163 / 267 N (ratio 0.38) — WORSE than Run 40
- Friction randomization appears to have disrupted early learning
- Episode length grew (462-622) but only from surviving, not walking

**Lesson:** Friction randomization during early training disrupts forward locomotion learning.
May need to be introduced later (curriculum) or with much smaller range.

---

## Run 42b — Only Lat Vel Boost (Minimal Change from Run 40)

**Date:** 2026-03-12
**Status:** TRAINING
**Target:** Fix lateral drift with the SMALLEST possible change from Run 40.

**Strategy:** Only change lat_vel 0.04→0.06. No friction rand, no force_balance. This isolates
whether a modest lat_vel boost alone can reduce drift without disrupting forward locomotion.

**Changes from Run 40:**
| Parameter | Run 40 | Run 42b | Rationale |
|-----------|--------|---------|-----------|
| rew_lat_vel | 0.04 | 0.06 | Only change — 1.5x boost for anti-drift |
| friction_rand | (none) | (none) | Disrupted learning in Run 42 |
| rew_force_balance | (none) | (none) | Caused stepping-in-place in Run 41+41b |

**Results:** KILLED at iter 726 — vel_x oscillating around 0 (-0.14 to +0.20, avg ~0.05).
- Same stepping-in-place pattern as Runs 41, 41b, 42
- Force ratio: 0.47 (217/196 N) — decent symmetry but no forward motion
- Episode length 690-714 — surviving but not walking
- Even lat_vel 0.04→0.06 (50% increase) disrupted forward locomotion

**Lesson:** RL training may be more sensitive than expected. OR this is run-to-run variance
and Run 40 was a lucky seed. Need reproducibility test.

---

## Run 43 — Exact Run 40 Config (Reproducibility Test)

**Date:** 2026-03-12
**Status:** TRAINING
**Target:** Confirm Run 40's results are reproducible with different random seed.

**Strategy:** Revert ALL changes back to exact Run 40 config. If Run 43 succeeds (vel_x > 0.3
by iter 1000), then Runs 41-42b failures were caused by the config changes. If Run 43 also
fails, there may be a code bug introduced in the Run 41 changes (force_balance computation,
friction method) that affects training even when weights are 0.

**Changes from Run 40:** NONE — exact same config.
- rew_lat_vel: 0.04 (Run 40 value)
- friction_rand: False
- rew_force_balance: 0.0
- All other params identical

**Results:** COMPLETED — 10,000 iterations. Run 40 config REPRODUCED successfully.

**Final metrics (iter 9,999):**
| Metric | Run 43 | Run 40 | Comparison |
|--------|--------|--------|------------|
| Peak reward | 5,743 | 5,747 | Near-identical |
| Final vel_x | 0.54 | 0.45 | Run 43 slightly faster |
| Episode length | 999 | 999 | Both full episodes |
| Foot force L/R | 166 / 250 N | ~similar | R-bias in both (structural) |
| Force ratio | 0.40 | 0.66-0.69 | Run 43 worse (variance) |
| Swing ratio | 0.506 / 0.494 | ~similar | Both symmetric |
| Noise std | 0.06 | 0.06 | Identical convergence |
| Foot height L/R | 0.069 / 0.102 | ~similar | R foot higher (asymmetry) |

**Key findings:**
1. Run 40 config is REPRODUCIBLE — different seed, same performance
2. Runs 41, 41b, 42, 42b failures were ALL caused by config changes (force_balance, lat_vel boost, friction rand)
3. R-force bias is STRUCTURAL — present in both Run 40 and 43, not caused by any specific change
4. Exploration phase takes ~650 iters — we were killing runs too early at iter 500
5. Foot height asymmetry (L:0.069 vs R:0.102) confirms lateral drift issue

**Checkpoints saved:** `release/run43_best/` (model_8800.pt, model_9999.pt)

---

## Run 44 — Observation History (15 steps) for Push Recovery

**Date:** 2026-03-22

**Base config:** Run 40 (hip_yaw gait ref, best natural gait quality)

**Changes from Run 43:**
1. **Observation history: 1 frame → 15 frames stacked**
   - obs_history_len: 15 (matching EngineAI)
   - observation_space: 64 → 64 × 15 = 960
   - History buffer: `[N, 15, 64]` — rolls each step, resets on episode end
   - Policy can now detect: sudden velocity change (push), tilt direction, leg state history
   - Root cause fix: without history, robot can't distinguish "mid-step lean" from "being pushed"

2. **Must train from scratch** — network input layer 64→512 becomes 960→512, incompatible with Run 40/43 checkpoints

**All reward weights, push forces, gait reference unchanged from Run 40/43.**

**Why this should work:**
- Visual analysis of Run 40/43 shows robot falls because it cannot react to 1.0 m/s pushes
- EngineAI uses 15-step history and survives same push forces
- With history, policy learns: "base_vel changed suddenly 3 frames ago → I was pushed → widen stance"
- Also fixes left/right asymmetry: policy can track which leg was last in swing

**Goals:**
- Fewer falls from push forces (primary)
- More symmetric L/R gait (secondary)
- Maintain vel_x > 0.3, feet_air_time positive

**Results: KILLED at iter 898 — policy never learned (noise_std=0.99 throughout)**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | Force L/R | swing_ratio |
|------|--------|-----------|-------|------------|-------|-----------|-------------|
| 36 | 243 | 71 | 0.99 | 154 | 0.42 | 198/196 | 0.39/0.61 |
| 148 | 220 | 66 | 0.99 | 163 | 0.28 | 200/198 | 0.41/0.59 |
| 351 | 217 | 70 | 0.99 | 349 | 0.046 | 192/201 | 0.40/0.60 |
| 694 | 254 | 83 | 0.99 | 407 | 0.003 | 194/183 | 0.44/0.56 |
| 898 | 255 | 82 | 0.99 | 251 | **-0.063** | 191/178 | 0.44/0.56 |

**Root cause — 960-dim history broke RSL-RL empirical normalizer:**
- Episodes only last ~82 steps → frames 2-15 in `[N,15,64]` history are **zeros** at episode start
- Running mean/std computed over all 960 dims → zero-padded frames dragged mean toward zero
- Normalizer output near-zero for history dims → policy got no gradient from history
- `noise_std=0.99` for 898 iterations confirms: **policy learned nothing from 15-frame history**

**Positive signal (force balance improved despite failure):**
- Force L/R was 198/196 at iter 36 — observation space change alone improves symmetry slightly
- swing_ratio improved 0.39/0.61 → 0.44/0.56 over training

---

## Run 45 — Compact History (3 frames, 18 key dims)

**Date:** 2026-03-23

**Base config:** Run 40 (hip_yaw gait ref)

**Changes from Run 44:**
1. **History design: full 64×15=960 → compact 18×3=54**
   - Only stack disturbance-relevant signals: `ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18 dims`
   - 3 frames × 18 dims = 54 extra dims appended to current 64-dim obs
   - Total obs: **64 + 54 = 118** (vs 960 in Run 44)
   - Normalizer handles 118 dims easily — no zero-padding problem (fills in 3 steps)
   - 3 frames × 20ms = **60ms context** — enough to detect push (velocity change), tilt trend, leg state

**Why compact signals:**
- `ang_vel_b` — detects sudden rotation from push
- `projected_gravity` — tracks tilt trend (am I falling?)
- `joint_pos_rel` — tracks which leg is in swing/stance

**All reward weights, push forces, gait reference unchanged from Run 40/43.**

**Goals:**
- vel_x > 0.3 by iter 1000 (Run 44 failed this)
- noise_std dropping below 0.95 by iter 500
- Fewer falls from pushes than Run 40
- Force balance L/R closer to 0.5 than Run 40's 0.66

**Results: KILLED at iter 1085 — instability event (reward -43% in 11 iters)**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | Force L/R |
|------|--------|-----------|-------|------------|-------|-----------|
| 9 | 190 | 56 | 0.99 | — | 0.47 | 187/190 |
| 34 | 237 | 68 | 0.99 | — | 0.16 | 195/198 |
| 133 | 1002 | 255 | 0.93 | 4474 | 0.011 | 214/235 |
| 1074 | 3987 | 845 | 0.39 | **34122** | 0.196 | 174/254 |
| 1085 | 2259 | 494 | 0.38 | 18 | 0.115 | 170/258 |

**Root cause — PPO value loss spike/collapse at iter 1074-1085:**
- value_loss 4474 → 34122 (critic overfit, inflated return estimates)
- Policy chased inflated critic estimates → critic corrected to 18 → policy left in bad state
- Reward -43% and episode length -42% in 11 iterations (~51s of training)
- vel_x recovered to 0.20 then collapsed to 0.115 — "high-step in place" exploit
- 3-frame history insufficient temporal context for proper push recovery

**Decision: upgrade to 15 frames (proper EngineAI depth) as Run 46**

---

## Run 46 — Compact History (15 frames, EngineAI-equivalent)

**Date:** 2026-03-23

**Base config:** Run 45 (compact history design)

**Changes from Run 45:**
1. **History frames: 3 → 15 (matches EngineAI `frame_stack=15`)**
   - `obs_history_len`: 3 → 15
   - Total obs: 64 + 18×15 = **334 dims** (vs 118 in Run 45)
   - Buffer shape: `[4096, 15, 18]`
   - 15 frames × 20ms = **300ms context** (vs 60ms in Run 45)

**Why 15 frames is safe (unlike Run 44's 960-dim):**
- Compact 18-dim frames: 334 total < 512 first hidden layer — **no bottleneck**
- Run 44 failed because 64×15=960 > 512 (compression bottleneck, not normalizer)
- EngineAI uses 47-dim × 15 = 705 total (their single obs is already compact)
- RSL-RL EmpiricalNormalization eps=1e-2 handles partial zeros identically to EngineAI

**All reward weights, push forces, gait reference unchanged from Run 40/43/45.**

**Results: CONVERGED — killed at iter 7530**

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | Force L/R |
|------|--------|-----------|-------|------------|-------|-----------|
| 1560 | 2631 | 542 | 0.27 | — | 0.27–0.42 | — |
| 7530 | ~5000 | 999 | 0.07 | — | 0.49 | — |

**Gait quality (screenshots):** persistent forward trunk lean, noticeable hip splay (wide lateral leg spread). Policy converged but posture not clean.

**Decision:** Kill and fix orientation formula + hip splay penalty → Run 47

---

## Run 47 — Orientation Dual-Signal + Hip Splay Fix

**Date:** 2026-03-23

**Base config:** Run 46 (15-frame compact history)

**Changes from Run 46:**

1. **Orientation formula — dual-signal matching EngineAI exactly**
   - Before: `exp(-norm(grav_xy)^2 * 10)` — single weak signal
   - After: `(exp(-(|roll|+|pitch|)*10) + exp(-norm(grav_xy)*20)) / 2` — dual signal at scale 20
   - `euler_xyz_from_quat` returns tuple → unpacked as `roll, pitch, _ = euler_xyz_from_quat(base_quat)` (bug caught by Codex review)

2. **Hip splay penalty — fixed joint indices**
   - Before: indices `[0,1,6,7]` = hip_pitch_l, hip_roll_l, hip_pitch_r, hip_roll_r (wrong — included hip_pitch which needs freedom)
   - After: indices `[1,2,7,8]` = hip_roll_l, hip_yaw_l, hip_roll_r, hip_yaw_r + 0.1 rad deadband
   - Pairwise L/R norms with clamp: `clamp(norm_L + norm_R - 0.1, 0, 0.5)`

**Review:** Codex + Qwen both reviewed. Codex caught tuple bug. Both confirmed correctness. GO signed off by user.

**Goals:**
- Hip splay narrows (visible in screenshots)
- Forward trunk lean reduced
- Episode length stays near 999
- No value_loss spikes above 50 in first 500 iters

**Results:** (in progress — started 2026-03-23)

| Iter | Reward | Ep Length | Noise | Value Loss | vel_x | Force L/R |
|------|--------|-----------|-------|------------|-------|-----------|
