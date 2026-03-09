# PM01 Bipedal Walking — Isaac Lab RL

Reinforcement learning project to teach the [PM01 humanoid robot](https://pollen-robotics.com/) to walk using NVIDIA Isaac Lab and RSL-RL (PPO).

## Overview

This project trains a bipedal walking policy for the PM01 robot's lower body (12 DOF legs) using:

- **Isaac Lab** — GPU-accelerated parallel simulation (4096 envs)
- **RSL-RL** — PPO implementation from RSL (Robotic Systems Lab, ETH Zurich)
- **Direct RL workflow** — custom environment without manager abstractions
- **EngineAI reference** — gait design and reward structure adapted from EngineAI's PM01 walking implementation

The robot learns to:
- Walk forward at commanded velocities (0.3–1.0 m/s)
- Track lateral and yaw commands
- Stand still when commanded
- Maintain balance under external pushes
- Follow a sinusoidal bipedal gait reference

## Project Structure

```
ArmRobotLegging/
├── source/ArmRobotLegging/ArmRobotLegging/
│   ├── tasks/direct/armrobotlegging/
│   │   ├── armrobotlegging_env.py        # Environment implementation
│   │   ├── armrobotlegging_env_cfg.py    # All hyperparameters & reward scales
│   │   ├── agents/
│   │   │   └── rsl_rl_ppo_cfg.py         # PPO algorithm config
│   │   └── __init__.py                   # Gym registration
│   └── robots/
│       ├── pm01.py                       # Robot articulation config (PD gains, init pose)
│       └── pm01_assets/urdf/             # URDF files
├── docs/
│   ├── TRAINING_FLOW.md                  # Architecture, step-by-step flow, hyperparameter reference
│   ├── TRAINING_HISTORY.md               # All training runs, results, and lessons learned
│   └── ENGINEAI_VS_ISAACLAB.md           # Gap analysis vs EngineAI reference implementation
├── Makefile                              # Training/play commands
├── CLAUDE.md                             # AI researcher standing orders
└── logs/                                 # Training checkpoints (gitignored)
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Gait reference joints | hip_pitch + knee + ankle | hip_yaw caused 360 spinning without obs history |
| Contact detection | Foot body z-height < 0.16m | Sensor-free; ankle body origin is 0.148m above ground |
| Swing penalty | Curriculum -1.5 → -0.8 | Static penalty can't solve stepping-vs-survival tradeoff |
| Commands | Forward-only (min 0.3 m/s) | Eliminates standing-still exploit that plagued 8 runs |
| Domain randomization | Push forces + PD gains ±20% | Forces reactive stepping and actuator robustness |

## Quick Start

### Prerequisites

- NVIDIA Isaac Lab (with Isaac Sim)
- Docker (recommended) or local Isaac Lab install

### Docker (recommended)

```bash
# Start the Isaac Lab container
docker compose up -d

# Enter the container
docker exec -it isaac-lab-base bash

# Inside container:
cd /workspace/armrobot/ArmRobotLegging
make install
make train-headless    # Train without GUI (~94K steps/s)
make train             # Train with GUI (~30K steps/s)
```

### Training

```bash
# Full training (4096 envs, 10000 iterations, ~6 hours headless)
make train-headless

# Debug training (64 envs, for testing changes)
make train-small

# Resume from checkpoint
make resume
```

### Visualize a Trained Policy

```bash
# Play latest checkpoint
make play-latest

# Play specific checkpoint
make play CHECKPOINT=/path/to/model_XXXX.pt
```

## Training Summary

| Run | Key Change | vel_x | feet_air_time | Ep Length | Outcome |
|-----|-----------|-------|---------------|-----------|---------|
| 1-12 | Various reward tuning | 0.0 | 0.0 | 999 | Standing still exploit |
| 13 | Forward-only commands | 0.76 | 0.0 | 400 | First walking (shuffle) |
| 14 | Rewards /5 | 0.60 | 0.0 | 794 | Stable shuffle, value loss fixed |
| 15 | Contact fix + swing penalty -0.4 | 0.51 | 0.04 | 709 | Too weak penalty |
| 16 | Swing penalty -1.5 | 0.73 | 0.62 | 88 | First stepping! Falls in 1.8s |
| 17 | Swing -0.8 + survival boost | 0.72 | 0.02 | 798 | Regresses to shuffling |
| 18 | Curriculum -1.5→-0.8 + PD rand | — | — | — | In progress |

See [docs/TRAINING_HISTORY.md](docs/TRAINING_HISTORY.md) for full details on all 18 runs.

## Reward Structure (22 terms)

**Positive rewards** (clamped sum >= 0): velocity tracking, gait reference, feet air time, contact pattern, orientation, base height, alive bonus, and more.

**Penalties** (always applied): action smoothness, energy, feet clearance, foot slip, termination, swing phase ground (curriculum), joint velocity/acceleration.

See [docs/TRAINING_FLOW.md](docs/TRAINING_FLOW.md) for the complete hyperparameter reference with formulas and purposes.

## Robot Specs

- **Model:** PM01 (legs only, 12 DOF)
- **Joints:** hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll (x2 legs)
- **PD gains:** hip/knee Kp=50-70, ankle Kp=20 (low — causes vibration challenges)
- **Standing height:** 0.8132m
- **Init pose:** 0.9m, knees bent (-0.24/0.48/-0.24 rad hip/knee/ankle)

## Documentation

- **[TRAINING_FLOW.md](docs/TRAINING_FLOW.md)** — Complete architecture, execution order, all 55 hyperparameters with purposes
- **[TRAINING_HISTORY.md](docs/TRAINING_HISTORY.md)** — Every training run, what changed, results, and lessons learned
- **[ENGINEAI_VS_ISAACLAB.md](docs/ENGINEAI_VS_ISAACLAB.md)** — Feature gap analysis vs EngineAI reference

## License

BSD-3-Clause
