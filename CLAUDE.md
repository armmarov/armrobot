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
- **Docs:** `docs/TRAINING_HISTORY.md`, `docs/TRAINING_FLOW.md`, `docs/PPO_NETWORK.md`, `docs/ENGINEAI_VS_ISAACLAB.md`
- **Training logs:** `/workspace/armrobot/logs/` (inside docker)
- **Makefile:** `Makefile` (train, train-headless, play-latest targets)

## Docker

- Container name: `isaac-lab-base`
- Training: `docker exec isaac-lab-base bash -c "cd /workspace/armrobot/ArmRobotLegging && make train-headless"`
- Play: `docker exec isaac-lab-base bash -c "cd /workspace/armrobot/ArmRobotLegging && make play-latest"`

## Convergence Criteria

- Kill training when reward plateaus for 500+ iterations with episode length near max (~1000)
- Run visual evaluation via `make play-latest` after killing training
- Record video evaluation and analyze frames if robot behavior is wrong

## Code Review & Git

- After fixing/updating code, **do a code review** before running training
- If code review passes, **commit and push to git** for tracking
- Commit message format: `Run N: <brief description of changes>`
- Push to the current branch after each run's code changes are verified
