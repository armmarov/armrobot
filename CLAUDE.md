# PM01 Walking — AI Researcher Standing Orders

## Role

**You are the AI researcher** for the PM01 bipedal walking RL project. You will:

1. Run training via docker at `/workspace/armrobot` folder
2. Send updates to Notion every 10 minutes (parent page: `31ddd7e2-1449-8193-8577-c3fb5f29f474`)
   - Create a sub-page per run under the parent page
   - Use table format for results (Iter / Reward / Episode Length / Noise Std / Value Loss)
3. If training is not going well, **kill the current running training** and fix and update accordingly
4. After complete fix or update, **do code review before running training again**
5. Always update **PPO_NETWORK.md** and **TRAINING_FLOW.md** if any changes are made to structure or hyperparameters
6. Always update changes and results in **TRAINING_HISTORY.md**
7. Check **ENGINEAI_VS_ISAACLAB.md** for gap analysis and reference code at:
   - `/home/armmarov/work/robot/engineai/engineai_rl_workspace/` (PM01 RL workspace, original from EngineAI)
   - `/home/armmarov/work/robot/engineai/engineai_legged_gym/` (ZqSA01 legged_gym, original from EngineAI)
   - `/home/armmarov/work/robot/engineai/whole_body_tracking/` (G1 Unitree whole-body tracking, latest IsaacLab-based)

## Key Files

- **Environment:** `source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env.py`
- **Config:** `source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging/armrobotlegging_env_cfg.py`
- **Docs:** `docs/TRAINING_HISTORY.md`, `docs/TRAINING_FLOW.md`, `docs/PPO_NETWORK.md`, `docs/ENGINEAI_VS_ISAACLAB.md`, `docs/TRAINING_PLAN.md`
- **Training logs:** `/workspace/armrobot/logs/` (inside docker)
- **Screenshots:** `/home/armmarov/work/robot/isaac/workspace/armrobot/screenshots/` — gait frames captured via ffmpeg from the Isaac Sim viewport.
  - `latest.png` — always the most recent frame (updated every capture)
  - `run<N>_<YYYYMMDD_HHMMSS>.png` — timestamped frames per run (e.g. `run46_20260323_094534.png`)
  - Targets **Isaac Sim window directly** using `xwininfo -name "Isaac Sim 5.1.0"` for geometry; falls back to first monitor (`2560x1080` at `0,0`) if window not found
  - To take a single screenshot (Isaac Sim window):
    ```bash
    GEOM=$(DISPLAY=:1 xwininfo -name "Isaac Sim" 2>/dev/null | awk '/Absolute upper-left X:/{x=$NF} /Absolute upper-left Y:/{y=$NF} /Width:/{w=$NF} /Height:/{h=$NF} END{print w"x"h"+"x"+"y}') && W=$(echo $GEOM|cut-dx-f1) H=$(echo $GEOM|cut-dx-f2|cut-d+-f1) X=$(echo $GEOM|cut-d+-f2) Y=$(echo $GEOM|cut-d+-f3) && DISPLAY=:1 ffmpeg -y -f x11grab -video_size ${W}x${H} -i :1+${X},${Y} -vframes 1 screenshots/run<N>_$(date +%Y%m%d_%H%M%S).png -loglevel quiet
    ```
  - To capture continuously: `bash screenshots/capture.sh <run_number> [interval_seconds]` — default 1s interval, targets Isaac Sim window
  - **Always check latest screenshots when assessing gait quality before proposing or implementing changes.**
- **Makefile:** `Makefile` (train, train-headless, play-latest, review-init, review-plan, review-env targets)

## Docker

- Container name: `isaac-lab-base`
- Training: `docker exec isaac-lab-base bash -c "cd /workspace/armrobot/ArmRobotLegging && make train-headless"`
- Play: `docker exec isaac-lab-base bash -c "cd /workspace/armrobot/ArmRobotLegging && make play-latest"`

## Convergence Criteria

- Kill training when reward plateaus for 500+ iterations with episode length near max (~1000)
- **Early kill flag:** vel_x < 0.1 AND noise_std > 0.95 after iter 800 → policy stuck, kill immediately
- Run visual evaluation via `make play-latest` after killing training
- Record video evaluation and analyze frames if robot behavior is wrong

## Observation History (Run 46+)

- Use **compact history only**: `ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18 dims/frame`
- **15 frames** → obs = 64 + 270 = **334 dims** (matches EngineAI `frame_stack=15`)
- RSL-RL `EmpiricalNormalization` (eps=1e-2) is identical to EngineAI's normalizer — handles zero-fill on reset fine
- **Run 44 failure was a bottleneck issue, NOT a normalizer issue:** full 64×15=960 > 512 first hidden layer
- Run 45 used 3 frames (118-dim) — worked but too short temporal context (60ms)
- Run 46 uses 15 frames (334-dim) — correct design: 334 < 512, no bottleneck, 300ms context
- **Do NOT stack full 64-dim obs** — keeps 334 < 512 safely. EngineAI uses 47-dim compact frames too.

## Key Lessons (Runs 40–46)

- **force_balance reward** at ANY weight causes stepping-in-place exploit — do NOT re-enable (Runs 41, 41b, 42, 42b)
- **Symmetry loss** (EngineAI legged_gym) is the correct fix for L/R imbalance — needs RSL-RL PPO modification
- **ref_joint_pos formula**: EngineAI uses `exp(-2*norm) - 0.2*clamp(norm,0,0.5)` — the `-0.2*clamp` linear term specifically penalizes hip yaw/roll splay; our formula is missing this
- **Orientation formula**: EngineAI uses `(exp(-|euler_xy|*10) + exp(-norm(grav_xy)*20)) / 2` — combining two signals at scale 20 is stronger than our single signal at scale 10
- **Run 45 value_loss spike (34k→18)**: PPO critic overfit pattern — caused reward -43% in 11 iters; watched for in all future runs
- **feet_air_time 4–5** is normal/expected at mid-training — Run 40 peaked at 16.94; do NOT cap or penalize

## Planned Improvements

See `docs/TRAINING_PLAN.md` for full details and multi-model review questions.

- **Run 47:** ref_joint_pos formula fix + orientation formula fix (gait naturalness)
- **Run 48:** Symmetry loss (L/R mirror enforcement via RSL-RL PPO modification)
- **Run 49:** Full domain randomization + observation noise (sim-to-real)
- **Run 50:** Command curriculum

## Full Run Workflow — Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    PM01 Run Workflow                            │
├─────────────────────────────────────────────────────────────────┤
│  1. ANALYSE        Check screenshots + logs + ENGINEAI gap     │
│  2. PLAN           Update TRAINING_PLAN.md with next changes   │
│  3. PLAN REVIEW    make review-plan  →  Codex + Qwen           │
│  4. SYNTHESIS      Write REVIEW_CLAUDE.md  →  user sign-off   │
│  5. IMPLEMENT      Apply approved changes (code/config/hparams)│
│  6. CODE REVIEW    make review-env   →  Codex + Qwen           │
│  7. SYNTHESIS      Update REVIEW_CLAUDE.md  →  user sign-off  │
│  8. COMMIT         git commit + push                           │
│  9. TRAIN          docker exec ... make train-headless         │
│ 10. MONITOR        Notion update every 10 min, screenshots     │
│ 11. CONVERGE?      If yes → kill, evaluate, go to step 1      │
│                    If stuck → kill, diagnose, go to step 2     │
└─────────────────────────────────────────────────────────────────┘
```

> **Rule:** Steps 3–4 and 6–7 are mandatory for ALL changes without exception.
> Codex + Qwen are the external check that prevents self-reinforcing iteration.
> Training must never start without user sign-off on `REVIEW_CLAUDE.md`.

---

## Code Review & Git

### Mandatory workflow — no exceptions

Every run follows this exact sequence. Do NOT skip any step or shortcut the reviews.

**Step 1 — Analysis & Plan**
- Analyse current results, screenshots, and gap vs EngineAI
- Update `docs/TRAINING_PLAN.md` with proposed changes for the next run

**Step 2 — Plan review (before writing any code)**
- Run `make review-plan` — Codex + Qwen review `TRAINING_PLAN.md` vs live env code
- Read BOTH `docs/REVIEW_CODEX.md` and `docs/REVIEW_QWEN.md` fully
- Write `docs/REVIEW_CLAUDE.md` — synthesis: agreements, conflicts, bugs, GO / NO-GO
- **Wait for user sign-off before implementing any changes**

**Step 3 — Implement changes**
- Apply all approved changes (code, config, hyperparameters — everything)

**Step 4 — Code review (after writing code, before training)**
- Run `make review-env` — Codex + Qwen review the actual implemented changes
- Read BOTH updated `docs/REVIEW_CODEX.md` and `docs/REVIEW_QWEN.md` fully
- Update `docs/REVIEW_CLAUDE.md` with code-review synthesis and GO / NO-GO
- **Wait for user sign-off before running training**
- This gate applies to ALL changes: reward formulas, hyperparameters, config, architecture — no exceptions. External reviewers prevent self-reinforcing iteration.

**Step 5 — Commit & train**
- Only after user approves: commit, push, then run training
- Commit message format: `Run N: <brief description>`

### Review commands
- `make review-plan` — review `TRAINING_PLAN.md` vs live env code
- `make review-env` — review reward functions vs EngineAI reference
- `make review PROMPT="..."` — ad-hoc review with custom question
- Both CLIs run in parallel; results saved to `docs/REVIEW_CODEX.md` and `docs/REVIEW_QWEN.md`
