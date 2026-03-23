from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ArmRobotLegging.robots.pm01 import PM01_CFG


@configclass
class ArmrobotleggingEnvCfg(DirectRLEnvCfg):
    # ---------- timing ----------
    decimation = 4
    episode_length_s = 20.0

    # ---------- spaces ----------
    # single frame obs: base_lin_vel_b(3) + base_ang_vel_b(3) + projected_gravity_b(3) +
    #      joint_pos_rel(12) + joint_vel(12) + prev_actions(12) +
    #      commands(3) + gait_phase(2) + ref_joint_diff(12) + contact_mask(2) = 64
    # Run 46: compact history — 15 frames matching EngineAI frame_stack=15
    #   history signals: ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18 dims/frame
    #   history size: 18 × 15 = 270 extra dims
    #   total obs: 64 + 270 = 334 dims
    # Run 44 failed: full 64×15=960 — input > first hidden layer (512) caused bottleneck
    # Run 45 used 3 frames (118-dim) — safe but limited temporal context
    # Run 46: compact 18-dim × 15 frames = 334 total — no bottleneck (334 < 512), matches EngineAI design
    obs_history_len: int = 15           # Run 46: 15 frames (matches EngineAI frame_stack=15)
    obs_history_size: int = 18          # unchanged: ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12)
    action_space = 12
    observation_space = 64 + 18 * 15   # 334 = 64 current + 270 compact history
    state_space = 0

    # ---------- simulation — 200 Hz physics, 50 Hz policy ----------
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # ---------- terrain ----------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ---------- scene ----------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # ---------- robot ----------
    robot_cfg: ArticulationCfg = PM01_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # leg joints controlled by the policy (order matters — matches action indices)
    leg_joint_names: list = [
        "j00_hip_pitch_l",    # 0
        "j01_hip_roll_l",     # 1
        "j02_hip_yaw_l",      # 2
        "j03_knee_pitch_l",   # 3
        "j04_ankle_pitch_l",  # 4
        "j05_ankle_roll_l",   # 5
        "j06_hip_pitch_r",    # 6
        "j07_hip_roll_r",     # 7
        "j08_hip_yaw_r",      # 8
        "j09_knee_pitch_r",   # 9
        "j10_ankle_pitch_r",  # 10
        "j11_ankle_roll_r",   # 11
    ]

    # bodies used for contact / termination
    foot_body_names: list = ["link_ankle_roll_l", "link_ankle_roll_r"]
    knee_body_names: list = ["link_knee_pitch_l", "link_knee_pitch_r"]
    termination_contact_body_names: list = [
        "link_base",
        # knee/torso have no collision in legs-only simple collision URDF
    ]

    # ---------- action ----------
    action_scale: float = 0.55  # Run 39: 0.6→0.55 — slight reduction for stability (anti-wobble)

    # ---------- gait parameters ----------
    cycle_time: float = 0.8               # gait cycle duration [s] (matching EngineAI)
    target_joint_pos_scale: float = 0.26  # Run 20: revert to EngineAI value (0.17 broke stepping)

    # ---------- velocity command ranges ----------
    cmd_lin_vel_x_range: tuple = (0.3, 1.0)     # [m/s] forward only — eliminates standing-still exploit
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)    # [m/s] reduced lateral range
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)    # [rad/s] reduced yaw range
    cmd_resample_time_s: float = 8.0             # resample commands every N seconds
    cmd_still_ratio: float = 0.0                 # Run 40: no standing — 100% walking focus (standing = fixed motor mode)

    # ---------- domain randomization: push forces ----------
    push_robots: bool = True            # enable random pushes (velocity impulses)
    push_interval_s: float = 8.0        # Run 35: EngineAI value (was 15.0 — more frequent = better reactive stepping)
    max_push_vel_xy: float = 1.0        # Run 33: match EngineAI (was 0.5 — stronger pushes for resilience)
    max_push_ang_vel: float = 0.6       # Run 33: match EngineAI (was 0.4)

    # ---------- domain randomization: PD gains ----------
    pd_gains_rand: bool = True                    # Run 18: randomize stiffness/damping ±20%
    stiffness_multi_range: tuple = (0.8, 1.2)     # multiplier range for stiffness
    damping_multi_range: tuple = (0.8, 1.2)       # multiplier range for damping

    # ---------- termination ----------
    termination_height: float = 0.45    # Run 37: revert (0.65 caused standing still in Run 35+36 — too strict for exploration)
    base_height_target: float = 0.8132  # nominal standing height [m]

    # ---------- contact sensor (Run 31: force-based, replaces z-height) ----------
    # EngineAI uses contact_forces[:, foot_indices, 2] > 5.0 N
    # Z-height contact (threshold=0.16m) was unreliable: foot at 0.148m standing means
    # only 0.012m margin. Shuffling feet at 0.15m stayed "in contact" — no air-time penalty.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,  # update every physics step
        track_air_time=True,
    )
    contact_force_threshold: float = 5.0  # [N] — vertical force above this = contact (EngineAI value)
    contact_height_threshold: float = 0.16  # [m] — kept for legacy, no longer used for foot contact

    # ---------- curriculum: swing penalty annealing (Run 18) ----------
    swing_penalty_start: float = -1.5           # aggressive at start (forces stepping)
    swing_penalty_end: float = -0.8             # relaxed at end (allows survival)
    swing_curriculum_steps: int = 144000        # anneal over ~3000 iters (3000 * 48 steps)

    # ---------- domain randomization: friction ----------
    friction_rand: bool = False                   # Run 42b: DISABLED — disrupted early learning in Run 42 (vel_x stuck ~0.05)
    friction_range: tuple = (0.7, 1.3)            # multiplier range for static+dynamic friction

    # ---------- reward scales (Run 43: exact Run 40 config, reproducibility test) ----------
    # Run 41-42b: ALL changes from Run 40 caused stepping-in-place.
    # Run 43: revert to EXACT Run 40 config to confirm baseline still works.
    #
    # velocity tracking — FULL EngineAI (Run 36: boost to drive forward walking)
    rew_tracking_lin_vel: float = 1.4        # Run 36: FULL EngineAI (was 0.93) — stronger forward drive
    rew_tracking_ang_vel: float = 1.1        # Run 36: FULL EngineAI (was 0.73) — match velocity tracking
    rew_tracking_sigma: float = 1.0          # Run 48: 5.0 too lenient — standing still at 0.3 m/s cost only 0.025 reward; 1.0 gives 5× stronger signal

    # gait quality — ref_joint_pos reverted, orientation kept full
    rew_ref_joint_pos: float = 1.47          # Run 36: REVERT to /1.5 (2.2 caused standing still in Run 35)
    rew_feet_air_time: float = 1.0           # Run 34: EngineAI 1.5 / 1.5 (keep)
    rew_feet_contact_number: float = 0.93    # Run 34: EngineAI 1.4 / 1.5 (keep)
    rew_orientation: float = 1.0             # Run 35: FULL EngineAI — stronger upright incentive (keep)
    rew_base_height: float = 0.2             # Run 35: FULL EngineAI — maintain standing height (keep)
    rew_feet_clearance: float = -1.6          # Run 38: FULL EngineAI (was -1.07 — caused shuffling)
    rew_default_joint_pos: float = 0.53      # Run 34: EngineAI 0.8 / 1.5 (keep)
    rew_feet_distance: float = 0.13          # Run 34: EngineAI 0.2 / 1.5 (keep)

    # feet distance limits [m]
    min_feet_dist: float = 0.15
    max_feet_dist: float = 0.8
    target_feet_height: float = 0.06         # Run 40: match original robot (3-5cm clearance) — was 0.12
    max_feet_height: float = 0.10            # Run 40: tighter range for natural steps — was 0.18
    rew_feet_height_max: float = -0.6        # Run 38: FULL EngineAI (was -0.4) — penalize too-low feet

    # penalties — stability-critical boosted, rest /1.5
    rew_action_smoothness: float = -0.003    # Run 39: FULL EngineAI (was -0.002 — stronger smoothness for anti-wobble)
    rew_energy: float = -0.000067            # Run 34: EngineAI -0.0001 / 1.5 (keep)
    rew_vel_mismatch: float = 0.5            # Run 35: FULL EngineAI (was 0.33) — penalize unwanted z-motion
    rew_foot_slip: float = -0.067            # Run 34: EngineAI -0.1 / 1.5 (keep)
    rew_alive: float = 0.0                   # not in EngineAI
    rew_termination: float = -0.0            # EngineAI uses -0.0
    rew_track_vel_hard: float = 0.5          # Run 35: FULL EngineAI (was 0.33) — force velocity tracking
    rew_low_speed: float = 0.2               # Run 35: FULL EngineAI (was 0.13) — punish slowness
    rew_dof_vel: float = -6.7e-6             # Run 34: EngineAI -1e-5 / 1.5 (keep)
    rew_dof_acc: float = -3.3e-9             # Run 34: EngineAI -5e-9 / 1.5 (keep)
    rew_lat_vel: float = 0.04                # Run 43: revert to Run 40 value (0.06 may have caused 42b failure)
    rew_swing_phase_ground: float = 0.0      # DISABLED (not in EngineAI)

    # --- Run 39: new anti-wobble rewards from EngineAI ---
    rew_base_acc: float = 0.2               # EngineAI: exp(-norm(base_acc) * 3) — rewards smooth base motion
    rew_knee_distance: float = 0.2          # EngineAI: prevents knee collision, keeps legs spaced
    target_knee_dist: float = 0.25          # target distance between knees [m]

    # --- Run 41/41b: force_balance REMOVED — caused stepping-in-place at any weight ---
    rew_force_balance: float = 0.0          # Run 42: DISABLED (0.08 and 0.15 both caused stepping-in-place)
