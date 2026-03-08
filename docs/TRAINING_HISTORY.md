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
| — | — | — | — | — | — | — |

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
