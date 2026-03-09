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

**Goals:**
- feet_air_time MUST become positive (this is the make-or-break metric)
- vel_x > 0.3
- Value loss < 200 (epochs=2 should eliminate spikes)
- This run tests whether contact detection was the root cause of shuffling
