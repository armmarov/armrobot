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
            A_NORM["Running Mean/Std Normalizer<br/>[64] → [64]"]
            A_L1["Linear(64, 512) + ELU"]
            A_L2["Linear(512, 256) + ELU"]
            A_L3["Linear(256, 128) + ELU"]
            A_OUT["Linear(128, 12)"]
            A_DIST["Gaussian Distribution<br/>μ = network output<br/>σ = learnable log_std<br/>(init σ = 1.0)"]
            A_NORM --> A_L1 --> A_L2 --> A_L3 --> A_OUT --> A_DIST
        end

        subgraph CRITIC["Critic (Value) Network"]
            direction TB
            C_NORM["Running Mean/Std Normalizer<br/>[64] → [64]"]
            C_L1["Linear(64, 512) + ELU"]
            C_L2["Linear(512, 256) + ELU"]
            C_L3["Linear(256, 128) + ELU"]
            C_OUT["Linear(128, 1)"]
            C_NORM --> C_L1 --> C_L2 --> C_L3 --> C_OUT
        end

        ACTION["Action [4096, 12]<br/>sampled from N(μ, σ²)"]
        VALUE["Value V(s) [4096, 1]<br/>estimated return"]

        A_DIST --> ACTION
        C_OUT --> VALUE
    end

    OBS --> A_NORM
    OBS --> C_NORM
    ACTION -->|"apply to env"| ENV
    ENV --> OBS
    ENV --> REW
    ENV --> DONE

    subgraph BUFFER["Rollout Buffer (24 steps)"]
        BUF_OBS["observations [24, 4096, 64]"]
        BUF_ACT["actions [24, 4096, 12]"]
        BUF_REW["rewards [24, 4096]"]
        BUF_VAL["values [24, 4096]"]
        BUF_LOG["log_probs [24, 4096]"]
        BUF_DONE["dones [24, 4096]"]
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
        I["Observation [64]<br/>────────────────<br/>lin_vel_b [3]<br/>ang_vel_b [3]<br/>proj_gravity [3]<br/>joint_pos_rel [12]<br/>joint_vel [12]<br/>prev_actions [12]<br/>commands [3]<br/>sin/cos_phase [2]<br/>ref_joint_diff [12]<br/>contact_mask [2]"]
    end

    subgraph NORMALIZE["Normalize"]
        N["x̂ = (x - μ_run) / σ_run<br/>────────────────<br/>Updated each iteration<br/>from collected obs"]
    end

    subgraph HIDDEN["Hidden Layers"]
        H1["Dense 512<br/>────────<br/>W: [64 × 512]<br/>b: [512]<br/>→ ELU"]
        H2["Dense 256<br/>────────<br/>W: [512 × 256]<br/>b: [256]<br/>→ ELU"]
        H3["Dense 128<br/>────────<br/>W: [256 × 128]<br/>b: [128]<br/>→ ELU"]
    end

    subgraph OUTPUT["Output"]
        MU["Mean μ [12]<br/>────────<br/>W: [128 × 12]<br/>b: [12]<br/>(no activation)"]
        SIGMA["Log Std [12]<br/>────────<br/>Learnable parameter<br/>init = log(1.0) = 0<br/>shared across states"]
        SAMPLE["Action = μ + σ × ε<br/>ε ~ N(0, 1)<br/>clamp to [-1, 1]"]
    end

    I --> N --> H1 --> H2 --> H3 --> MU --> SAMPLE
    SIGMA --> SAMPLE
```

## Critic Network Detail

```mermaid
flowchart LR
    subgraph INPUT2["Input"]
        I2["Observation [64]<br/>(same as actor)"]
    end

    subgraph NORMALIZE2["Normalize"]
        N2["x̂ = (x - μ_run) / σ_run<br/>(separate normalizer)"]
    end

    subgraph HIDDEN2["Hidden Layers"]
        H2_1["Dense 512 → ELU"]
        H2_2["Dense 256 → ELU"]
        H2_3["Dense 128 → ELU"]
    end

    subgraph OUTPUT2["Output"]
        V["Value V(s) [1]<br/>────────<br/>W: [128 × 1]<br/>b: [1]<br/>(no activation)<br/><br/>Estimates total<br/>discounted return<br/>from state s"]
    end

    I2 --> N2 --> H2_1 --> H2_2 --> H2_3 --> V
```

## PPO Update Algorithm

```mermaid
flowchart TB
    subgraph GAE["Step 1: Compute Advantages"]
        G1["For t = T-1 down to 0:<br/>δ_t = r_t + γ V(s_{t+1}) - V(s_t)<br/>A_t = δ_t + γλ A_{t+1}<br/><br/>γ = 0.99 (discount)<br/>λ = 0.95 (GAE smoothing)"]
        G2["Returns = Advantages + Values"]
        G1 --> G2
    end

    subgraph EPOCHS["Step 2: Optimize (5 epochs)"]
        direction TB
        SHUFFLE["Shuffle all 98304 transitions<br/>(24 steps × 4096 envs)"]

        subgraph MB["Step 3: Mini-batch (4 batches of 24576)"]
            direction TB
            RATIO["Compute probability ratio:<br/>r(θ) = π_new(a|s) / π_old(a|s)"]

            subgraph LOSSES["Compute Losses"]
                direction LR
                PL["Policy Loss<br/>L_clip = -min(<br/>  r(θ) × A,<br/>  clip(r(θ), 1±0.2) × A<br/>)"]
                VL["Value Loss<br/>L_value = 1.0 × max(<br/>  (V - R)²,<br/>  (V_clip - R)²<br/>)"]
                EL["Entropy Bonus<br/>L_entropy = -0.005 × H(π)<br/><br/>H = 0.5 × log(2πeσ²)<br/>Encourages exploration"]
            end

            TOTAL["Total Loss = L_clip + L_value + L_entropy"]
            GRAD["Backpropagate<br/>Clip gradient norm ≤ 1.0"]
            STEP["Adam optimizer step<br/>lr = 1e-3 (adaptive)"]

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
            CP1["Linear(64, 512):  64×512 + 512  = 33,280"]
            CP2["Linear(512, 256): 512×256 + 256 = 131,328"]
            CP3["Linear(256, 128): 256×128 + 128 = 32,896"]
            CP4["Linear(128, 1):   128×1 + 1     = 129"]
            CP_TOTAL["Critic Total: 197,633"]
        end

        GRAND["Grand Total: ~396,697 parameters"]
    end
```

## Training Timeline

```mermaid
gantt
    title PM01 Walking Training (3000 iterations)
    dateFormat X
    axisFormat %s

    section Data Collection
    24 steps × 4096 envs    :a1, 0, 24

    section PPO Update
    GAE computation          :a2, after a1, 1
    Epoch 1 (4 mini-batches) :a3, after a2, 4
    Epoch 2 (4 mini-batches) :a4, after a3, 4
    Epoch 3 (4 mini-batches) :a5, after a4, 4
    Epoch 4 (4 mini-batches) :a6, after a5, 4
    Epoch 5 (4 mini-batches) :a7, after a6, 4

    section Next Iteration
    Collect again            :a8, after a7, 24
```

## How Action Noise Evolves

```mermaid
flowchart LR
    subgraph EARLY["Early Training (iter 0-500)"]
        E1["σ ≈ 1.0 (high)<br/>Actions are nearly random<br/>Robot falls constantly<br/>Exploring joint space"]
    end
    subgraph MID["Mid Training (iter 500-1500)"]
        E2["σ ≈ 0.3-0.5<br/>Robot learns to stand<br/>Begins tracking commands<br/>Gait pattern emerging"]
    end
    subgraph LATE["Late Training (iter 1500-3000)"]
        E3["σ ≈ 0.1-0.2 (low)<br/>Stable walking gait<br/>Tracks velocity commands<br/>Fine-tuning smoothness"]
    end
    EARLY --> MID --> LATE
```

## Summary Table

| Component | Shape | Parameters | File |
|-----------|-------|------------|------|
| Actor input | [64] | - | `armrobotlegging_env_cfg.py` |
| Actor hidden | [512, 256, 128] | 197,504 | `rsl_rl_ppo_cfg.py` |
| Actor output (μ) | [12] | 1,548 | `rsl_rl_ppo_cfg.py` |
| Actor log_std | [12] | 12 | RSL-RL (init_noise_std=1.0) |
| Critic hidden | [512, 256, 128] | 197,504 | `rsl_rl_ppo_cfg.py` |
| Critic output (V) | [1] | 129 | RSL-RL |
| Obs normalizer | μ,σ [64] each | - (not trainable) | RSL-RL (empirical) |
| **Total trainable** | | **~396,697** | |
| Rollout buffer | 24 × 4096 | 98,304 transitions | `rsl_rl_ppo_cfg.py` |
| Training iterations | 3000 | ~295M env steps | `rsl_rl_ppo_cfg.py` |
