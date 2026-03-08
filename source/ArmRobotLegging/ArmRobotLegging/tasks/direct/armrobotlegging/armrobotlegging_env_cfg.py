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
    target_joint_pos_scale: float = 0.26  # amplitude of reference gait [rad]

    # ---------- velocity command ranges ----------
    cmd_lin_vel_x_range: tuple = (0.3, 1.0)     # [m/s] forward only — eliminates standing-still exploit
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)    # [m/s] reduced lateral range
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)    # [rad/s] reduced yaw range
    cmd_resample_time_s: float = 8.0             # resample commands every N seconds
    cmd_still_ratio: float = 0.0                 # no zero commands (was 0.1 — exploited by standing)

    # ---------- domain randomization: push forces ----------
    push_robots: bool = True            # enable random pushes (velocity impulses)
    push_interval_s: float = 4.0        # push every N seconds (more frequent than EngineAI)
    max_push_vel_xy: float = 1.0        # max linear velocity impulse [m/s] (EngineAI base=1.0)
    max_push_ang_vel: float = 0.8       # max angular velocity impulse [rad/s]

    # ---------- termination ----------
    termination_height: float = 0.45    # reset if base z < this [m]
    base_height_target: float = 0.8132  # nominal standing height [m]

    # ---------- contact thresholds ----------
    contact_height_threshold: float = 0.03  # [m] — foot z-height below this = contact

    # ---------- reward scales ----------
    # velocity tracking
    rew_tracking_lin_vel: float = 1.4
    rew_tracking_ang_vel: float = 1.1
    rew_tracking_sigma: float = 2.5

    # gait quality
    rew_ref_joint_pos: float = 2.2       # follow gait reference
    rew_feet_air_time: float = 1.5       # reward proper swing duration
    rew_feet_contact_number: float = 1.4  # reward proper contact pattern per phase
    rew_orientation: float = 1.0         # stay upright
    rew_base_height: float = 0.2         # maintain target height
    rew_feet_clearance: float = -1.6     # penalise low swing foot (error norm)
    rew_default_joint_pos: float = 0.8   # keep hip pitch/roll near default
    rew_feet_distance: float = 0.2       # penalise feet too close or too far

    # feet distance limits [m]
    min_feet_dist: float = 0.15
    max_feet_dist: float = 0.8
    target_feet_height: float = 0.1      # swing foot target height [m]

    # penalties
    rew_action_smoothness: float = -0.003  # penalise jerk
    rew_energy: float = -0.0001            # penalise torque * velocity
    rew_vel_mismatch: float = 0.5          # penalise z-vel and xy-ang-vel
    rew_foot_slip: float = -0.1            # penalise foot sliding during contact
    rew_alive: float = 0.05               # small survival bonus (keep robot upright)
    rew_termination: float = -1.0          # fall penalty (discourage falling)
    rew_track_vel_hard: float = 0.5        # sharp velocity tracking (exp(-error*10))
    rew_low_speed: float = 1.5             # discrete: -1 too slow, +2 good, -2 wrong dir (7.5× vs Run 11)
    rew_dof_vel: float = -1e-5             # penalise joint velocities (prevents vibration)
    rew_dof_acc: float = -5e-9             # penalise joint accelerations (CRITICAL anti-vibration)
    rew_lat_vel: float = 0.3              # lateral velocity tracking (prevents sideways drift)
