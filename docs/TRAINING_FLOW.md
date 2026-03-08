# PM01 Walking — Training Flow

## High-Level Training Loop

```mermaid
flowchart TB
    subgraph INIT["INITIALIZATION (once)"]
        A1["1. Launch Isaac Sim<br/><b>train.py</b>"]
        A2["2. gym.make('Isaac-PM01-Walking-Direct-v0')<br/>triggers <b>__init__.py</b> registration"]
        A3["3. Create ArmrobotleggingEnv<br/><b>armrobotlegging_env.py → __init__()</b>"]
        A4["4. _setup_scene()<br/>Spawn 4096 PM01 robots + ground<br/><b>armrobotlegging_env.py</b>"]
        A5["5. Create PPO Runner<br/>Actor [512,256,128] + Critic [768,256,128]<br/><b>rsl_rl_ppo_cfg.py</b>"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph LOOP["TRAINING LOOP (10000 iterations)"]
        subgraph COLLECT["Collect Experience (48 steps per env)"]
            B1["6. Policy outputs action<br/>[4096, 12] floats in [-1, 1]"]
            B2["7. _pre_physics_step(action)<br/>Clamp & store action"]
            B3["8. _apply_action() × 4 (decimation)<br/>target = default_pos + 0.5 × action<br/>→ set_joint_position_target()"]
            B4["9. Physics steps (4 × 0.005s = 0.02s)"]
            B5["10. _get_dones()<br/>Update gait phase + contacts + commands<br/>Check termination"]
            B6["11. _get_rewards()<br/>Compute 21 reward terms → scalar + per-term log"]
            B7["12. _reset_idx(fallen_envs)<br/>Reset state + resample commands"]
            B8["13. _get_observations()<br/>Build 64-dim obs tensor"]
            B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8
            B8 -->|"repeat 48×"| B1
        end

        subgraph UPDATE["PPO Update"]
            C1["14. Compute advantages (GAE)<br/>γ=0.994, λ=0.9"]
            C2["15. Update policy (2 epochs, 4 mini-batches)<br/>lr=1e-5, clip=0.2, entropy=0.001"]
            C3["16. Update observation normalizer<br/>(empirical running stats)"]
        end

        COLLECT --> UPDATE
        UPDATE -->|"next iteration"| COLLECT
    end

    subgraph SAVE["CHECKPOINTING"]
        D1["17. Save model every 200 iterations<br/>logs/rsl_rl/pm01_walking/"]
    end

    INIT --> LOOP
    UPDATE --> SAVE
```

## Detailed Step-by-Step with Parameters

### Phase 1: Action Generation

```mermaid
flowchart LR
    subgraph POLICY["PPO Actor Network"]
        direction TB
        OBS["Observation [4096, 64]"]
        NORM["Normalize<br/>(empirical running stats)"]
        L1["Linear(64, 512) + ELU"]
        L2["Linear(512, 256) + ELU"]
        L3["Linear(256, 128) + ELU"]
        L4["Linear(128, 12) → μ [4096, 12]"]
        NOISE["Training: μ + σ × ε (ε ~ N(0,1))<br/>Deployment: μ only<br/>(init σ=1.0, decays via learning)"]
        ACTION["Action [4096, 12]<br/>clamped to [-1, 1]"]
        OBS --> NORM --> L1 --> L2 --> L3 --> L4 --> NOISE --> ACTION
    end

    subgraph FILES1["Files"]
        F1["<b>rsl_rl_ppo_cfg.py</b><br/>actor_hidden_dims=[512,256,128]<br/>activation=elu<br/>init_noise_std=1.0"]
    end
```

### Phase 2: Action Application (per physics sub-step)

```mermaid
flowchart LR
    subgraph APPLY["_apply_action() — called 4× per policy step"]
        ACTION["action [4096, 12]<br/>in [-1, 1]"]
        SCALE["× action_scale<br/>(0.5 rad)"]
        DEFAULT["+ default_joint_pos<br/>(bent knee standing)"]
        TARGET["Joint position target"]
        PD["ImplicitActuator PD Controller<br/>τ = Kp(target - pos) - Kd × vel"]
        PHYSICS["PhysX simulation<br/>dt = 0.005s"]
        ACTION --> SCALE --> DEFAULT --> TARGET --> PD --> PHYSICS
    end

    subgraph FILES2["Files & Parameters"]
        F2a["<b>armrobotlegging_env_cfg.py</b><br/>action_scale = 0.5<br/>decimation = 4<br/>sim.dt = 1/200"]
        F2b["<b>pm01.py</b><br/>hip_pitch: Kp=70, Kd=7<br/>hip_roll/yaw: Kp=50, Kd=5<br/>knee: Kp=70, Kd=7<br/>ankle: Kp=20, Kd=0.2"]
    end
```

### Phase 3: State Update & Gait Phase

```mermaid
flowchart TB
    subgraph DONES["_get_dones() — runs BEFORE rewards"]
        direction TB
        GP["_update_gait_phase()<br/>phase = (step × dt × dec / 0.8) % 1<br/>phase[still_commands] = 0 (freeze when standing)<br/>sin_phase = sin(2π × phase)<br/>ref_pos: hip_pitch(0/6) + knee(3/9) + ankle(4/10)<br/>amplitudes: 0.26/0.52/0.26 rad<br/>deadband: ref=0 when |sin|<0.05"]
        FC["_update_foot_contact()<br/>contact = foot_z < 0.03m<br/>first_contact = contact & !last_contact<br/>air_time_on_contact = air_time × first_contact<br/>air_time reset on contact"]
        CMD["_update_commands()<br/>Every 400 steps: resample vx, vy, yaw_rate<br/>10% chance zero command"]
        TERM["Termination check<br/>fell = base_z < 0.45m<br/>bad_contact = base on ground<br/>(legs-only URDF: knee/torso have no collision)"]
        GP --> FC --> CMD --> TERM
    end

    subgraph FILES3["Files & Parameters"]
        F3["<b>armrobotlegging_env_cfg.py</b><br/>cycle_time = 0.8s<br/>target_joint_pos_scale = 0.26<br/>termination_height = 0.45<br/>contact_height_threshold = 0.03<br/>cmd_resample_time_s = 8.0<br/>cmd_still_ratio = 0.1"]
    end
```

### Phase 4: Observation Construction

```mermaid
flowchart LR
    subgraph OBS["_get_observations() → [4096, 64]"]
        direction TB
        O1["lin_vel_b [3]<br/>Body-frame linear velocity"]
        O2["ang_vel_b [3]<br/>Body-frame angular velocity"]
        O3["projected_gravity [3]<br/>Gravity in body frame"]
        O4["joint_pos_rel [12]<br/>Current - default joint pos"]
        O5["joint_vel [12]<br/>Joint velocities"]
        O6["prev_actions [12]<br/>Last policy output"]
        O7["commands [3]<br/>Target vx, vy, yaw_rate"]
        O8["sin_phase [1] + cos_phase [1]<br/>Gait clock"]
        O9["ref_joint_diff [12]<br/>Reference gait - current pos"]
        O10["contact_mask [2]<br/>Left/right foot on ground"]
    end

    subgraph FILES4["File"]
        F4["<b>armrobotlegging_env.py</b><br/>_get_observations()"]
    end
```

### Phase 5: Reward Computation

```mermaid
flowchart TB
    subgraph REWARDS["compute_rewards() → scalar per env"]
        direction TB

        subgraph POS["Positive rewards (clamped ≥ 0)"]
            R1["tracking_lin_vel = 1.4 × exp(-error²/5.0)<br/>Match commanded vx, vy"]
            R2["tracking_ang_vel = 1.1 × exp(-error²/5.0)<br/>Match commanded yaw rate"]
            R3["ref_joint_pos = 2.2 × exp(-2 × mean(diff²))<br/>Follow gait reference (MEAN over joints)"]
            R4["feet_air_time = 1.5 × Σ(air_time - 0.5) × first_contact<br/>Reward proper swing duration<br/>(gated: zero when cmd < 0.1)"]
            R5["feet_contact_number = 1.4 × mean(match)<br/>Correct stance/swing per phase"]
            R6["orientation = 1.0 × exp(-roll_pitch_err × 10)<br/>Stay upright"]
            R7["base_height = 0.2 × exp(-height_err × 100)<br/>Maintain 0.8132m"]
            R8["vel_mismatch = 0.5 × (low_z_vel + low_xy_angvel)<br/>Minimize parasitic motion"]
            R9["alive = 0.05<br/>Survival bonus"]
            R12["default_joint_pos = 0.8 × (exp(-hip_dev×100) - 0.01×norm)<br/>Keep hip pitch/roll near default"]
            R13["feet_distance = 0.2 × exp(-deviation×100)<br/>Keep feet within [0.15m, 0.8m]"]
            R_TVH["track_vel_hard = 0.5 × (exp(-err×10) - 0.2×err)<br/>Sharp velocity tracking (forces locomotion)"]
            R_LS["low_speed = 0.2 × discrete(-1/+2/-2)<br/>Punish too slow, reward good speed"]
            R_LV["lat_vel = 0.3 × exp(-lat_error²×10)<br/>Lateral velocity tracking"]
        end

        CLAMP["torch.clamp(sum, min=0.0)"]

        subgraph NEG["Penalties (always applied)"]
            R10["action_smoothness = -0.003 × (term1 + term2 + term3)<br/>term1: (a_t - a_{t-1})²<br/>term2: (a_t + a_{t-2} - 2×a_{t-1})² (2nd-order)<br/>term3: 0.05 × |a_t|"]
            R11["energy = -0.0001 × Σ(action² × |vel|)<br/>Efficiency"]
            R14["feet_clearance = -1.6 × norm(swing_target - foot_height)<br/>Force swing foot to lift"]
            R15["foot_slip = -0.1 × Σ(√foot_speed × contact)<br/>Penalize sliding on ground"]
            R16["termination = -1.0 × fell<br/>Fall penalty"]
            R17["track_vel_hard = 0.5 × (exp(-err×10) - 0.2×err)<br/>Sharp velocity tracking"]
            R18["low_speed = 0.2 × discrete(-1/+2/-2)<br/>Punish too slow, reward good speed"]
            R19["dof_vel = -1e-5 × Σ(joint_vel²)<br/>Penalize joint velocities (anti-vibration)"]
            R20["dof_acc = -5e-9 × Σ((Δvel/dt)²)<br/>Penalize joint accelerations (CRITICAL anti-vibration)"]
            R21["lat_vel = 0.3 × exp(-lat_error² × 10)<br/>Lateral velocity tracking (prevents sideways drift)"]
        end

        POS --> CLAMP --> NEG
    end

    subgraph FILES5["Files & Parameters"]
        F5["<b>armrobotlegging_env_cfg.py</b><br/>All rew_* parameters<br/><b>armrobotlegging_env.py</b><br/>compute_rewards()"]
    end
```

### Phase 6: PPO Update

```mermaid
flowchart TB
    subgraph PPO["PPO Update Phase"]
        direction TB
        BUF["Rollout buffer filled<br/>48 steps × 4096 envs = 196,608 transitions"]
        GAE["Compute GAE advantages<br/>γ = 0.994, λ = 0.9"]
        subgraph EPOCHS["2 epochs"]
            MINI["Split into 4 mini-batches<br/>(196,608 / 4 = 49,152 transitions each)"]
            LOSS["Compute Total Loss:<br/>Policy Loss (clip ratio ε=0.2, uses advantages)<br/>+ Value Loss (coef=1.0, uses returns)<br/>+ Entropy Loss (coef=0.001, uses σ)"]
            BACK["Single backpropagation:<br/>Policy Loss → Actor gradients<br/>Value Loss → Critic gradients<br/>Entropy Loss → log_std gradient"]
            GRAD["Gradient clip<br/>max_norm = 1.0"]
            LR["Adam optimizer step<br/>base lr = 1e-4<br/>adaptive via KL (target = 0.01)"]
            MINI --> LOSS --> BACK --> GRAD --> LR
        end
        BUF --> GAE --> EPOCHS
    end

    subgraph FILES6["File"]
        F6["<b>rsl_rl_ppo_cfg.py</b><br/>num_steps_per_env = 48<br/>gamma = 0.994, lam = 0.9<br/>clip_param = 0.2<br/>entropy_coef = 0.001<br/>learning_rate = 1e-5<br/>num_learning_epochs = 2<br/>num_mini_batches = 4<br/>max_grad_norm = 1.0<br/>desired_kl = 0.01"]
    end
```

## Complete Parameter Map

### Where every parameter lives

```
┌─────────────────────────────────────────────────────────────────┐
│                        train.py                                  │
│  --task Isaac-PM01-Walking-Direct-v0                            │
│  --num_envs 4096                                                 │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              __init__.py (gym registration)               │  │
│  │  entry_point → ArmrobotleggingEnv                         │  │
│  │  env_cfg     → ArmrobotleggingEnvCfg                      │  │
│  │  agent_cfg   → PM01WalkingPPORunnerCfg                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────┐  ┌────────────────────────────────┐   │
│  │  armrobotlegging_    │  │  rsl_rl_ppo_cfg.py             │   │
│  │  env_cfg.py          │  │                                │   │
│  │                      │  │  Network:                      │   │
│  │  SIMULATION:         │  │    [512, 256, 128] ELU         │   │
│  │    dt = 1/200        │  │    noise_std = 1.0             │   │
│  │    decimation = 4    │  │    obs normalization = True    │   │
│  │                      │  │                                │   │
│  │  SPACES:             │  │  PPO:                          │   │
│  │    action = 12       │  │    γ = 0.994, λ = 0.9         │   │
│  │    obs = 64          │  │    lr = 1e-5 (adaptive)        │   │
│  │                      │  │    clip = 0.2                  │   │
│  │  GAIT:               │  │    entropy = 0.001             │   │
│  │    cycle = 0.8s      │  │    epochs = 2                  │   │
│  │    scale = 0.26 rad  │  │    mini-batches = 4            │   │
│  │                      │  │    steps/env = 48              │   │
│  │  COMMANDS:            │  │    max_iterations = 10000      │   │
│  │    vx: [-1, 1] m/s   │  │                                │   │
│  │    vy: [-0.3, 0.3]   │  └────────────────────────────────┘   │
│  │    yaw: [-1, 1] rad/s│                                       │
│  │    resample: 8s       │  ┌────────────────────────────────┐   │
│  │    still: 10%         │  │  pm01.py (robot config)        │   │
│  │                      │  │                                │   │
│  │  REWARDS:             │  │  PD Gains:                     │   │
│  │    lin_vel: 1.4       │  │    hip_pitch: Kp=70, Kd=7     │   │
│  │    ang_vel: 1.1       │  │    knee: Kp=70, Kd=7          │   │
│  │    ref_pos: 2.2       │  │    ankle: Kp=20, Kd=0.2       │   │
│  │    air_time: 1.5      │  │                                │   │
│  │    contact: 1.4       │  │  Effort limits:                │   │
│  │    orient: 1.0        │  │    hip: 164 Nm                 │   │
│  │    height: 0.2        │  │    knee: 164 Nm                │   │
│  │    vel_mis: 0.5       │  │    ankle: 52 Nm                │   │
│  │    alive: 0.05        │  │                                │   │
│  │    smooth: -0.003     │  │  Init: 0.9m, knees bent        │   │
│  │    energy: -0.0001    │  │  URDF: pm01_only_legs_simple_   │   │
│  │        collision.urdf           │   │
│  │    clearance: -1.6    │  │                                │   │
│  │    default_pos: 0.8   │  │                                │   │
│  │    feet_dist: 0.2     │  │                                │   │
│  │    foot_slip: -0.1    │  │                                │   │
│  │    term: -1.0         │  │                                │   │
│  │    track_hard: 0.5    │  │                                │   │
│  │    low_speed: 0.2     │  │                                │   │
│  │    dof_vel: -1e-5     │  │                                │   │
│  │    dof_acc: -5e-9     │  │                                │   │
│  │    lat_vel: 0.3       │  └────────────────────────────────┘   │
│  │                      │                                       │
│  │  TERMINATION:         │                                       │
│  │    height < 0.45m     │                                       │
│  │    body contact       │                                       │
│  │    timeout: 20s       │                                       │
│  └─────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

## IsaacLab Step Execution Order

```mermaid
sequenceDiagram
    participant P as PPO Policy
    participant E as ArmrobotleggingEnv
    participant S as PhysX Simulator
    participant R as Robot (PM01)

    Note over P,R: Each training step (50 Hz)

    P->>E: action [4096, 12]
    E->>E: _pre_physics_step()<br/>save prev_actions, clamp action<br/>apply push forces (every 8s)

    loop 4× (decimation)
        E->>R: _apply_action()<br/>set_joint_position_target()
        R->>S: PD torques applied
        S->>S: Simulate 0.005s
        S->>R: Update joint/body states
    end

    E->>E: _get_dones()<br/>├─ _update_gait_phase()<br/>├─ _update_foot_contact()<br/>├─ _update_commands()<br/>└─ check termination

    E->>E: _get_rewards()<br/>compute 21 reward terms<br/>accumulate per-term episode sums

    E->>E: _reset_idx(fallen_envs)<br/>log per-term rewards to extras["log"]<br/>reset state + new commands

    E->>E: _get_observations()<br/>build 64-dim tensor

    E->>P: obs [4096, 64], reward [4096], done [4096]

    Note over P: After 48 steps: PPO update
```
