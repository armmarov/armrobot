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
    # obs: base_lin_vel_b(3) + base_ang_vel_b(3) + projected_gravity_b(3) +
    #      joint_pos_rel(12) + joint_vel(12) + prev_actions(12) +
    #      commands(3) + gait_phase(2) + ref_joint_diff(12) + contact_mask(2) = 64
    action_space = 12
    observation_space = 64
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
    termination_contact_body_names: list = [
        "link_base",
        # knee/torso have no collision in legs-only simple collision URDF
    ]

    # ---------- action ----------
    action_scale: float = 0.5  # [rad] — policy output in [-1,1] * scale + default_pos

    # ---------- gait parameters ----------
    cycle_time: float = 0.8               # gait cycle duration [s] (matching EngineAI)
    target_joint_pos_scale: float = 0.26  # Run 20: revert to EngineAI value (0.17 broke stepping)

    # ---------- velocity command ranges ----------
    cmd_lin_vel_x_range: tuple = (0.3, 1.0)     # [m/s] forward only — eliminates standing-still exploit
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)    # [m/s] reduced lateral range
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)    # [rad/s] reduced yaw range
    cmd_resample_time_s: float = 8.0             # resample commands every N seconds
    cmd_still_ratio: float = 0.1                 # Run 18: re-enable standing (10% zero commands)

    # ---------- domain randomization: push forces ----------
    push_robots: bool = True            # enable random pushes (velocity impulses)
    push_interval_s: float = 15.0       # Run 33: match EngineAI (was 5.0 — too frequent, limited episode to ~81%)
    max_push_vel_xy: float = 1.0        # Run 33: match EngineAI (was 0.5 — stronger pushes for resilience)
    max_push_ang_vel: float = 0.6       # Run 33: match EngineAI (was 0.4)

    # ---------- domain randomization: PD gains ----------
    pd_gains_rand: bool = True                    # Run 18: randomize stiffness/damping ±20%
    stiffness_multi_range: tuple = (0.8, 1.2)     # multiplier range for stiffness
    damping_multi_range: tuple = (0.8, 1.2)       # multiplier range for damping

    # ---------- termination ----------
    termination_height: float = 0.45    # reset if base z < this [m]
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

    # ---------- reward scales (Run 33: full EngineAI weights) ----------
    # Run 32 proved walking works with /2.5 scaling + biped fixes.
    # Run 33: upgrade to full EngineAI weights — stronger learning signal now that
    # core formulas (biped air-time, stance_mask, force-based contact) are correct.
    # Run 29 failed with full weights because formulas were still broken (quadruped air-time,
    # z-height contact). Now safe to use full weights.
    #
    # velocity tracking
    rew_tracking_lin_vel: float = 1.4        # Run 33: full EngineAI weight
    rew_tracking_ang_vel: float = 1.1        # Run 33: full EngineAI weight
    rew_tracking_sigma: float = 5.0          # EngineAI value (not a weight)

    # gait quality — full EngineAI weights
    rew_ref_joint_pos: float = 2.2           # Run 33: full EngineAI weight
    rew_feet_air_time: float = 1.5           # Run 33: full EngineAI weight
    rew_feet_contact_number: float = 1.4     # Run 33: full EngineAI weight
    rew_orientation: float = 1.0             # Run 33: full EngineAI weight
    rew_base_height: float = 0.2             # Run 33: full EngineAI weight
    rew_feet_clearance: float = -1.6         # Run 33: full EngineAI weight
    rew_default_joint_pos: float = 0.8       # Run 33: full EngineAI weight
    rew_feet_distance: float = 0.2           # Run 33: full EngineAI weight

    # feet distance limits [m]
    min_feet_dist: float = 0.15
    max_feet_dist: float = 0.8
    target_feet_height: float = 0.10         # EngineAI value
    max_feet_height: float = 0.15            # margin above target
    rew_feet_height_max: float = -0.6        # Run 33: full EngineAI weight

    # penalties — full EngineAI weights
    rew_action_smoothness: float = -0.003    # Run 33: full EngineAI weight
    rew_energy: float = -0.0001              # Run 33: full EngineAI weight
    rew_vel_mismatch: float = 0.5            # Run 33: full EngineAI weight
    rew_foot_slip: float = -0.1              # Run 33: full EngineAI weight
    rew_alive: float = 0.0                   # not in EngineAI
    rew_termination: float = -0.0            # EngineAI uses -0.0
    rew_track_vel_hard: float = 0.5          # Run 33: full EngineAI weight
    rew_low_speed: float = 0.2               # Run 33: full EngineAI weight
    rew_dof_vel: float = -1e-5               # Run 33: full EngineAI weight
    rew_dof_acc: float = -5e-9               # Run 33: full EngineAI weight
    rew_lat_vel: float = 0.06                # Run 33: full weight (ours only)
    rew_swing_phase_ground: float = 0.0      # DISABLED (not in EngineAI)
