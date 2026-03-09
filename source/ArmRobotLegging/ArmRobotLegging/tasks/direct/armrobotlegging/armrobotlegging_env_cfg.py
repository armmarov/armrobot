from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
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
    push_interval_s: float = 5.0        # push every N seconds (gentler — let robot learn stepping first)
    max_push_vel_xy: float = 0.5        # max linear velocity impulse [m/s] (halved from Run 13)
    max_push_ang_vel: float = 0.4       # max angular velocity impulse [rad/s] (halved from Run 13)

    # ---------- domain randomization: PD gains ----------
    pd_gains_rand: bool = True                    # Run 18: randomize stiffness/damping ±20%
    stiffness_multi_range: tuple = (0.8, 1.2)     # multiplier range for stiffness
    damping_multi_range: tuple = (0.8, 1.2)       # multiplier range for damping

    # ---------- termination ----------
    termination_height: float = 0.45    # reset if base z < this [m]
    base_height_target: float = 0.8132  # nominal standing height [m]

    # ---------- contact thresholds ----------
    # NOTE: link_ankle_roll body origin is ~0.148m above ground when standing flat.
    # Previous value of 0.03m meant contact was NEVER detected (broken since Run 1).
    contact_height_threshold: float = 0.16  # [m] — foot z-height below this = contact

    # ---------- curriculum: swing penalty annealing (Run 18) ----------
    swing_penalty_start: float = -1.5           # aggressive at start (forces stepping)
    swing_penalty_end: float = -0.8             # relaxed at end (allows survival)
    swing_curriculum_steps: int = 144000        # anneal over ~3000 iters (3000 * 48 steps)

    # ---------- reward scales (Run 25: /2.5 base + targeted gait boost + fix free rewards) ----------
    # Run 23: /2.5 stable + forward walking but shuffling (feet_air_time=0)
    # Run 24: boosting all 4 gait terms to /1 destabilized value loss (ref_joint_pos
    #         gives ~800 free reward from joints near zero). Fix: only boost the 3
    #         terms that directly enforce stepping, reduce free reward sources.
    #
    # velocity tracking — tighten sigma to reduce free reward from standing
    rew_tracking_lin_vel: float = 0.56       # /2.5
    rew_tracking_ang_vel: float = 0.44       # /2.5
    rew_tracking_sigma: float = 3.0          # Run 25: tighten (was 5.0 — gave 95% reward for standing)

    # gait quality — only 3 stepping enforcers boosted (NOT ref_joint_pos)
    rew_ref_joint_pos: float = 0.88          # Run 25: back to /2.5 (2.2 gave 800 free reward)
    rew_feet_air_time: float = 1.5           # FULL EngineAI — directly rewards foot lifting
    rew_feet_contact_number: float = 1.4     # FULL EngineAI — forces alternating gait
    rew_orientation: float = 0.40            # /2.5
    rew_base_height: float = 0.08            # /2.5
    rew_feet_clearance: float = -1.6         # FULL EngineAI — penalizes shuffling
    rew_default_joint_pos: float = 0.05      # Run 25: near-zero (was 0.32 — exp(-100x) penalized stepping)
    rew_feet_distance: float = 0.08          # /2.5

    # feet distance limits [m]
    min_feet_dist: float = 0.15
    max_feet_dist: float = 0.8
    target_feet_height: float = 0.10         # keep EngineAI value (not scaled — geometry param)
    max_feet_height: float = 0.15            # keep (not in EngineAI)
    rew_feet_height_max: float = -0.24       # -0.6 / 2.5

    # penalties
    rew_action_smoothness: float = -0.0012   # EngineAI -0.003 / 2.5
    rew_energy: float = -0.00002             # keep (different formula than EngineAI)
    rew_vel_mismatch: float = 0.20           # EngineAI 0.5 / 2.5
    rew_foot_slip: float = -0.04             # EngineAI -0.1 / 2.5
    rew_alive: float = 0.03                  # keep (not in EngineAI)
    rew_termination: float = -0.0            # keep (EngineAI uses -0.0)
    rew_track_vel_hard: float = 0.20         # EngineAI 0.5 / 2.5
    rew_low_speed: float = 0.08              # EngineAI 0.2 / 2.5
    rew_dof_vel: float = -4e-6               # EngineAI -1e-5 / 2.5
    rew_dof_acc: float = -2e-9               # EngineAI -5e-9 / 2.5
    rew_lat_vel: float = 0.06                # keep (not in EngineAI)
    rew_swing_phase_ground: float = -0.8     # keep curriculum (not scaled — it's our addition)
