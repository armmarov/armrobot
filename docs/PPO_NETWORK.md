# PPO Network Architecture — PM01 Walking

## Full PPO System

```mermaid
flowchart TB
    subgraph ENV["Environment (4096 parallel)"]
        OBS["Observation<br/>[4096, 64]"]
        REW["Reward<br/>[4096]"]
        DONE["Done<br/>[4096]"]
    end

    subgraph AGENT["PPO Agent"]
        subgraph ACTOR["Actor (Policy) Network"]
            direction TB
            A_NORM["Running Mean/Std Normalizer<br/>[4096, 64] → [4096, 64]"]
            A_L1["Linear(64, 512) + ELU<br/>→ [4096, 512]"]
            A_L2["Linear(512, 256) + ELU<br/>→ [4096, 256]"]
            A_L3["Linear(256, 128) + ELU<br/>→ [4096, 128]"]
            A_OUT["Linear(128, 12)<br/>→ μ [4096, 12]"]
            A_DIST["Gaussian Distribution<br/>μ = Linear(128, 12) output<br/>σ = exp(log_std), log_std is a<br/>separate learnable parameter [12]<br/>(init log_std = 0 → σ = 1.0)"]
            A_NORM --> A_L1 --> A_L2 --> A_L3 --> A_OUT --> A_DIST
        end

        subgraph CRITIC["Critic (Value) Network"]
            direction TB
            C_NORM["Running Mean/Std Normalizer<br/>[4096, 64] → [4096, 64]"]
            C_L1["Linear(64, 768) + ELU<br/>→ [4096, 768]"]
            C_L2["Linear(768, 256) + ELU<br/>→ [4096, 256]"]
            C_L3["Linear(256, 128) + ELU<br/>→ [4096, 128]"]
            C_OUT["Linear(128, 1)<br/>→ V [4096, 1]"]
            C_NORM --> C_L1 --> C_L2 --> C_L3 --> C_OUT
        end

        EPSILON["ε ~ N(0, 1) [4096, 12]<br/>random noise (torch.randn)<br/>training only"]
        ACTION["Action [4096, 12]<br/>Training: μ + σ × ε<br/>Deployment: μ only"]
        VALUE["Value V(s) [4096, 1]<br/>estimated return"]

        A_DIST --> ACTION
        EPSILON -->|"training only"| ACTION
        C_OUT --> VALUE
    end

    OBS --> A_NORM
    OBS --> C_NORM
    ACTION -->|"apply to env"| ENV
    ENV --> OBS
    ENV --> REW
    ENV --> DONE

    subgraph BUFFER["Rollout Buffer (48 steps)"]
        BUF_OBS["observations [48, 4096, 64]"]
        BUF_ACT["actions [48, 4096, 12]"]
        BUF_REW["rewards [48, 4096]"]
        BUF_VAL["values [48, 4096]"]
        BUF_LOG["log_probs [48, 4096]"]
        BUF_DONE["dones [48, 4096]"]
    end

    OBS --> BUF_OBS
    ACTION --> BUF_ACT
    REW --> BUF_REW
    VALUE --> BUF_VAL
    A_DIST -->|"log π(a|s)"| BUF_LOG
    DONE --> BUF_DONE
```

## Actor Network Detail

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        I["Observation [4096, 64]<br/>────────────────<br/>lin_vel_b [3]<br/>ang_vel_b [3]<br/>proj_gravity [3]<br/>joint_pos_rel [12]<br/>joint_vel [12]<br/>prev_actions [12]<br/>commands [3]<br/>sin/cos_phase [2]<br/>ref_joint_diff [12]<br/>contact_mask [2]"]
    end

    subgraph NORMALIZE["Normalize"]
        N["x̂ = (x - μ_run) / σ_run<br/>────────────────<br/>Updated each iteration<br/>from collected obs"]
    end

    subgraph HIDDEN["Hidden Layers"]
        H1["Dense 512<br/>────────<br/>W: [64 × 512]<br/>b: [512]<br/>→ ELU<br/>out: [4096, 512]"]
        H2["Dense 256<br/>────────<br/>W: [512 × 256]<br/>b: [256]<br/>→ ELU<br/>out: [4096, 256]"]
        H3["Dense 128<br/>────────<br/>W: [256 × 128]<br/>b: [128]<br/>→ ELU<br/>out: [4096, 128]"]
    end

    subgraph OUTPUT["Output"]
        MU["Mean μ [4096, 12]<br/>────────<br/>W: [128 × 12]<br/>b: [12]<br/>(no activation)"]
        SIGMA["Log Std [12]<br/>────────<br/>Learnable parameter<br/>init = log(1.0) = 0<br/>shared across states"]
        SAMPLE["Training:<br/>Action [4096, 12] = μ + σ × ε<br/>ε ~ N(0, 1)<br/>────────<br/>Deployment:<br/>Action [4096, 12] = μ<br/>(no noise, deterministic)"]
        CLAMP["Clamp to [-1, 1]<br/>────────<br/>Action [4096, 12]<br/>within valid range"]
    end

    I --> N --> H1 --> H2 --> H3 --> MU --> SAMPLE --> CLAMP
    SIGMA -->|"training only"| SAMPLE
```

## Critic Network Detail

```mermaid
flowchart LR
    subgraph INPUT2["Input"]
        I2["Observation [4096, 64]<br/>(same as actor)"]
    end

    subgraph NORMALIZE2["Normalize"]
        N2["x̂ = (x - μ_run) / σ_run<br/>(separate normalizer)"]
    end

    subgraph HIDDEN2["Hidden Layers"]
        H2_1["Dense 768 → ELU<br/>out: [4096, 768]"]
        H2_2["Dense 256 → ELU<br/>out: [4096, 256]"]
        H2_3["Dense 128 → ELU<br/>out: [4096, 128]"]
    end

    subgraph OUTPUT2["Output"]
        V["Value V(s) [4096, 1]<br/>────────<br/>W: [128 × 1]<br/>b: [1]<br/>(no activation)<br/><br/>Predicts total future reward<br/>from current state (single float)"]
    end

    I2 --> N2 --> H2_1 --> H2_2 --> H2_3 --> V
```

## How Critic V(s) Connects to Actor Training

```mermaid
flowchart TB
    subgraph INPUTS["Inputs (from rollout buffer)"]
        direction LR
        ENV_R["r (reward)<br/>from environment<br/>[4096]"]
        CRITIC_V["V(s) (value of current state)<br/>from Critic network<br/>[4096]"]
        CRITIC_V2["V(s') (value of next state)<br/>from Critic network<br/>[4096]"]
    end

    subgraph COMPUTE["Computed (not learned, just math)"]
        direction TB
        ADV["Advantage: A = r + γ × V(s') − V(s)<br/>────────────────<br/>γ = 0.99<br/><br/>A > 0 → action was BETTER than expected<br/>A < 0 → action was WORSE than expected"]
        RET["Return: R = A + V(s)<br/>────────────────<br/>The 'true' value the critic<br/>should have predicted"]
    end

    subgraph USE["Who uses what"]
        direction LR
        subgraph ACTOR_USE["A → Updates ACTOR"]
            AU["Policy Loss uses A to decide:<br/>• A > 0: increase action probability<br/>  (do this more often)<br/>• A < 0: decrease action probability<br/>  (avoid this action)"]
        end
        subgraph CRITIC_USE["R → Updates CRITIC"]
            CU["Value Loss = (V(s) − R)²<br/>────────────────<br/>Teaches critic to predict<br/>more accurately next time<br/><br/>If V(s) was too high → lower it<br/>If V(s) was too low → raise it"]
        end
    end

    ENV_R --> ADV
    CRITIC_V --> ADV
    CRITIC_V2 --> ADV
    ADV --> RET
    CRITIC_V --> RET
    ADV --> ACTOR_USE
    RET --> CRITIC_USE
```

## PPO Update Algorithm

```mermaid
flowchart TB
    subgraph NUMBERS["Key Numbers"]
        direction LR
        N1["num_envs = 4096<br/>num_steps_per_env = 48<br/>num_mini_batches = 4<br/>num_learning_epochs = 2"]
        N2["total_transitions = 48 × 4096 = 196,608<br/>mini_batch_size = 196,608 / 4 = 49,152<br/>updates_per_iter = 4 batches × 2 epochs = 8"]
        N1 --> N2
    end

    subgraph GAE["Step 1: Compute Advantages"]
        G1["For t = T-1 down to 0:<br/>δ_t = r_t + γ V(s_{t+1}) - V(s_t)<br/>A_t = δ_t + γλ A_{t+1}<br/><br/>γ = 0.994 (discount)<br/>λ = 0.9 (GAE smoothing)"]
        G2["Returns = Advantages + Values"]
        G1 --> G2
    end

    subgraph EPOCHS["Step 2: Optimize (5 epochs)"]
        direction TB
        SHUFFLE["Shuffle all 196,608 transitions<br/>(48 steps × 4096 envs)"]

        subgraph MB["Step 3: Mini-batch (4 batches of 49,152)"]
            direction TB
            RATIO["Compute probability ratio:<br/>r(θ) = π_new(a|s) / π_old(a|s)"]

            subgraph LOSSES["Compute Losses"]
                direction LR
                PL["Policy Loss<br/>L_clip = -min(<br/>  r(θ) × A,<br/>  clip(r(θ), 1±0.2) × A<br/>)"]
                VL["Value Loss<br/>L_value = 1.0 × max(<br/>  (V - R)²,<br/>  (V_clip - R)²<br/>)"]
                EL["Entropy Bonus<br/>L_entropy = -0.001 × H(π)<br/><br/>H = 0.5 × log(2πeσ²)<br/>Encourages exploration"]
            end

            TOTAL["Total Loss = L_clip + L_value + L_entropy"]
            GRAD["Backpropagate<br/>Clip gradient norm ≤ 1.0"]
            STEP["Adam optimizer step<br/>lr = 1e-5 (adaptive)"]

            RATIO --> LOSSES --> TOTAL --> GRAD --> STEP
        end

        SHUFFLE --> MB
    end

    subgraph KL["Step 4: Adaptive Learning Rate"]
        KL1["Compute KL divergence<br/>between old and new policy"]
        KL2{"KL > 2 × 0.01?"}
        KL3["lr = lr / 1.5<br/>(too aggressive)"]
        KL4{"KL < 0.01 / 2?"}
        KL5["lr = lr × 1.5<br/>(too conservative)"]
        KL6["lr unchanged"]
        KL1 --> KL2
        KL2 -->|yes| KL3
        KL2 -->|no| KL4
        KL4 -->|yes| KL5
        KL4 -->|no| KL6
    end

    GAE --> EPOCHS --> KL
```

## Backpropagation & Gradient Descent

```mermaid
flowchart TB
    subgraph FORWARD["Forward Pass (compute losses)"]
        direction TB
        F_NOTE["Mini-batch size = 49,152<br/>= 196,608 total transitions / 4 mini-batches<br/>= (48 steps × 4096 envs) / 4"]
        F_OBS["Observations [49152, 64]<br/>(from mini-batch)"]
        F_ACT["Stored actions [49152, 12]<br/>(from rollout buffer)"]
        F_ADV["Advantages [49152]<br/>(from GAE, not learned)"]
        F_RET["Returns [49152]<br/>(from GAE, not learned)"]
        F_OLDLP["Old log_probs [49152]<br/>(from rollout buffer, frozen)"]

        subgraph RECOMPUTE["Recompute with current weights"]
            direction TB
            R_ACTOR["Actor forward pass<br/>obs → new μ, σ"]
            R_NEWLP["New log_prob = Σ log N(action | μ_new, σ_new)<br/>[49152]"]
            R_CRITIC["Critic forward pass<br/>obs → new V(s) [49152, 1]"]
            R_ACTOR --> R_NEWLP
        end

        F_OBS --> R_ACTOR
        F_OBS --> R_CRITIC
        F_ACT --> R_NEWLP
    end

    subgraph LOSS["Loss Computation"]
        direction TB
        L_RATIO["ratio = exp(new_log_prob - old_log_prob)<br/>How much policy changed"]
        L_POLICY["Policy Loss = -min(ratio × adv, clip(ratio) × adv)<br/>Uses: new_log_prob, old_log_prob, advantages"]
        L_VALUE["Value Loss = (V_new - returns)²<br/>Uses: critic output, returns"]
        L_ENTROPY["Entropy Loss = -0.001 × H(π)<br/>Uses: σ (encourages exploration)"]
        L_TOTAL["Total Loss = Policy + Value + Entropy"]
        L_RATIO --> L_POLICY --> L_TOTAL
        L_VALUE --> L_TOTAL
        L_ENTROPY --> L_TOTAL
    end

    subgraph BACKWARD["Backpropagation (compute gradients)"]
        direction TB
        B1["Total Loss → propagate backward<br/>through the computation graph"]
        subgraph ACTOR_GRAD["Policy Loss + Entropy Loss<br/>→ Actor Gradients"]
            direction TB
            AG_NOTE["Policy Loss flows through new_log_prob<br/>→ back through actor layers<br/>Entropy Loss flows through σ<br/>→ back to log_std"]
            AG1["∂Loss/∂(Actor W1, b1) [64×512 + 512]"]
            AG2["∂Loss/∂(Actor W2, b2) [512×256 + 256]"]
            AG3["∂Loss/∂(Actor W3, b3) [256×128 + 128]"]
            AG4["∂Loss/∂(Actor W4, b4) [128×12 + 12]"]
            AG5["∂Loss/∂(log_std) [12]"]
            AG_NOTE --> AG1
        end
        subgraph CRITIC_GRAD["Value Loss<br/>→ Critic Gradients"]
            direction TB
            CG_NOTE["Value Loss flows through V(s)<br/>→ back through critic layers"]
            CG1["∂Loss/∂(Critic W1, b1) [64×768 + 768]"]
            CG2["∂Loss/∂(Critic W2, b2) [768×256 + 256]"]
            CG3["∂Loss/∂(Critic W3, b3) [256×128 + 128]"]
            CG4["∂Loss/∂(Critic W4, b4) [128×1 + 1]"]
            CG_NOTE --> CG1
        end
        B1 --> ACTOR_GRAD
        B1 --> CRITIC_GRAD
    end

    subgraph CLIP_GRAD["Gradient Clipping"]
        GC["Clip total gradient norm ≤ 1.0<br/>Prevents exploding gradients"]
    end

    subgraph UPDATE["Gradient Descent (Adam Optimizer)"]
        direction TB
        U1["For each parameter θ:<br/>θ_new = θ - lr × adjusted_gradient<br/>────────────────<br/>Adam adjusts gradient using:<br/>• momentum (past gradient direction)<br/>• RMSprop (past gradient magnitude)<br/>lr = 1e-5 (adaptive)"]
        subgraph UPDATED["Updated Parameters (~478,873 total)"]
            direction LR
            UP1["Actor weights & biases<br/>(199,064 params)"]
            UP2["Actor log_std [12]<br/>(exploration noise)"]
            UP3["Critic weights & biases<br/>(279,809 params)"]
        end
        U1 --> UPDATED
    end

    FORWARD --> LOSS --> BACKWARD --> CLIP_GRAD --> UPDATE
    UPDATE -->|"repeat 4 mini-batches × 2 epochs = 8 updates per iteration"| FORWARD
```

### What drives the learning

| Signal | Source | Affects | How |
|--------|--------|---------|-----|
| **Advantages** | GAE from rewards + critic values | Actor weights | Positive advantage → increase action probability, negative → decrease |
| **Returns** | GAE from rewards | Critic weights | Critic learns to predict total future reward more accurately |
| **Entropy** | Current σ (log_std) | log_std parameter | Prevents σ from collapsing to 0 too early (keeps exploring) |
| **Gradient clipping** | All gradients | All weights | Caps update magnitude to prevent training instability |
| **Adaptive LR** | KL divergence | Learning rate | Slows down if policy changes too fast, speeds up if too slow |

### What is NOT learned (frozen during update)

| Value | Why frozen |
|-------|-----------|
| Old log_probs | Snapshot of policy at collection time — needed for ratio |
| Advantages | Computed once from rewards, not differentiable |
| Returns | Target for critic, computed from rewards |
| Actions in buffer | Already taken, can't change them |
| Observations | Came from environment, not from policy |

## Parameter Count

```mermaid
flowchart TB
    subgraph PARAMS["Total Trainable Parameters"]
        direction TB
        subgraph ACTOR_P["Actor"]
            AP1["Linear(64, 512):  64×512 + 512  = 33,280"]
            AP2["Linear(512, 256): 512×256 + 256 = 131,328"]
            AP3["Linear(256, 128): 256×128 + 128 = 32,896"]
            AP4["Linear(128, 12):  128×12 + 12   = 1,548"]
            AP5["log_std [12]:                      12"]
            AP_TOTAL["Actor Total: 199,064"]
        end

        subgraph CRITIC_P["Critic"]
            CP1["Linear(64, 768):  64×768 + 768  = 49,920"]
            CP2["Linear(768, 256): 768×256 + 256 = 196,864"]
            CP3["Linear(256, 128): 256×128 + 128 = 32,896"]
            CP4["Linear(128, 1):   128×1 + 1     = 129"]
            CP_TOTAL["Critic Total: 279,809"]
        end

        GRAND["Grand Total: ~478,873 parameters"]
    end
```

## Training Timeline

```mermaid
gantt
    title PM01 Walking Training (10000 iterations)
    dateFormat X
    axisFormat %s

    section Data Collection
    48 steps × 4096 envs    :a1, 0, 48

    section PPO Update (2 epochs)
    GAE computation          :a2, after a1, 1
    Epoch 1 (4 mini-batches) :a3, after a2, 4
    Epoch 2 (4 mini-batches) :a4, after a3, 4

    section Next Iteration
    Collect again            :a5, after a4, 48
```

## How Action Noise Evolves

```mermaid
flowchart LR
    subgraph EARLY["Early Training (iter 0-1000)"]
        E1["σ ≈ 1.0 (high)<br/>Actions are nearly random<br/>Robot falls constantly<br/>Exploring joint space"]
    end
    subgraph MID["Mid Training (iter 1000-5000)"]
        E2["σ ≈ 0.3-0.5<br/>Robot learns to stand<br/>Begins tracking commands<br/>Gait pattern emerging"]
    end
    subgraph LATE["Late Training (iter 5000-10000)"]
        E3["σ ≈ 0.1-0.2 (low)<br/>Stable walking gait<br/>Tracks velocity commands<br/>Fine-tuning smoothness"]
    end
    EARLY --> MID --> LATE
```

## Observation Vector Detail [64]

| Index | Dims | Name | Description | Source |
|-------|------|------|-------------|--------|
| 0–2 | 3 | `lin_vel_b` | Base linear velocity in body frame [m/s] | IMU / sim |
| 3–5 | 3 | `ang_vel_b` | Base angular velocity in body frame [rad/s] | IMU / sim |
| 6–8 | 3 | `projected_gravity` | Gravity vector in body frame (0,0,−1 when upright) | Quaternion transform |
| 9–20 | 12 | `joint_pos_rel` | Current leg joint positions minus default standing pose [rad] | Joint encoders |
| 21–32 | 12 | `joint_vel` | Leg joint velocities [rad/s] | Joint encoders |
| 33–44 | 12 | `prev_actions` | Previous policy output [−1, 1] | Action buffer |
| 45–47 | 3 | `commands` | Target velocity: [vx, vy, yaw_rate] | Command sampler |
| 48 | 1 | `sin_phase` | sin(2π × gait_phase) — gait clock | Gait generator |
| 49 | 1 | `cos_phase` | cos(2π × gait_phase) — gait clock | Gait generator |
| 50–61 | 12 | `ref_joint_diff` | Reference gait position minus current joint position [rad] | Gait generator |
| 62–63 | 2 | `contact_mask` | Left/right foot on ground [0 or 1] | Foot height check |
| | **64** | | | |

> Defined in `armrobotlegging_env.py` → `_get_observations()`
> Dimensions configured in `armrobotlegging_env_cfg.py` → `observation_space = 64`

## Action Vector Detail [12]

| Index | Joint | Side | Description |
|-------|-------|------|-------------|
| 0 | `j00_hip_pitch_l` | Left | Hip forward/backward |
| 1 | `j01_hip_roll_l` | Left | Hip side-to-side |
| 2 | `j02_hip_yaw_l` | Left | Hip rotation |
| 3 | `j03_knee_pitch_l` | Left | Knee bend |
| 4 | `j04_ankle_pitch_l` | Left | Ankle forward/backward |
| 5 | `j05_ankle_roll_l` | Left | Ankle side-to-side |
| 6 | `j06_hip_pitch_r` | Right | Hip forward/backward |
| 7 | `j07_hip_roll_r` | Right | Hip side-to-side |
| 8 | `j08_hip_yaw_r` | Right | Hip rotation |
| 9 | `j09_knee_pitch_r` | Right | Knee bend |
| 10 | `j10_ankle_pitch_r` | Right | Ankle forward/backward |
| 11 | `j11_ankle_roll_r` | Right | Ankle side-to-side |

Each action value is in **[−1, 1]**, converted to joint target:
```
target = default_joint_pos + 0.5 × action
```

> Defined in `armrobotlegging_env_cfg.py` → `leg_joint_names` and `action_scale = 0.5`
> Applied in `armrobotlegging_env.py` → `_apply_action()`

## Summary Table

| Component | Shape | Parameters | File |
|-----------|-------|------------|------|
| Actor input | [64] | - | `armrobotlegging_env_cfg.py` |
| Actor hidden | [512, 256, 128] | 197,504 | `rsl_rl_ppo_cfg.py` |
| Actor output (μ) | [12] | 1,548 | `rsl_rl_ppo_cfg.py` |
| Actor log_std | [12] | 12 | RSL-RL (init_noise_std=1.0) |
| Critic hidden | [768, 256, 128] | 279,680 | `rsl_rl_ppo_cfg.py` |
| Critic output (V) | [1] | 129 | RSL-RL |
| Obs normalizer | μ,σ [64] each | - (not trainable) | RSL-RL (empirical) |
| **Total trainable** | | **~478,873** | |
| Rollout buffer | 48 × 4096 | 196,608 transitions | `rsl_rl_ppo_cfg.py` |
| Training iterations | 10000 | ~1.97B env steps | `rsl_rl_ppo_cfg.py` |
