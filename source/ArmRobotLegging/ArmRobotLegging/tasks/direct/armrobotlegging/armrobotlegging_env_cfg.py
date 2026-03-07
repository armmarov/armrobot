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
        "link_knee_pitch_l",
        "link_knee_pitch_r",
        "link_torso_yaw",
    ]

    # ---------- action ----------
    action_scale: float = 0.5  # [rad] — policy output in [-1,1] * scale + default_pos

    # ---------- gait parameters ----------
    cycle_time: float = 0.64              # gait cycle duration [s]
    target_joint_pos_scale: float = 0.26  # amplitude of reference gait [rad]

    # ---------- velocity command ranges ----------
    cmd_lin_vel_x_range: tuple = (-1.0, 1.0)   # [m/s]
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)    # [m/s]
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)    # [rad/s]
    cmd_resample_time_s: float = 8.0             # resample commands every N seconds
    cmd_still_ratio: float = 0.1                 # probability of zero-velocity commands

    # ---------- termination ----------
    termination_height: float = 0.45    # reset if base z < this [m]
    base_height_target: float = 0.8132  # nominal standing height [m]

    # ---------- contact thresholds ----------
    foot_contact_threshold: float = 5.0   # [N] — forces above this = foot in contact
    contact_height_threshold: float = 0.03  # [m] — foot below this = contact

    # ---------- reward scales ----------
    # velocity tracking
    rew_tracking_lin_vel: float = 1.5
    rew_tracking_ang_vel: float = 1.0
    rew_tracking_sigma: float = 0.25

    # gait quality
    rew_ref_joint_pos: float = 2.0       # follow gait reference
    rew_feet_air_time: float = 1.5       # reward proper swing duration
    rew_feet_contact_number: float = 1.2  # reward proper contact pattern per phase
    rew_orientation: float = 1.0         # stay upright
    rew_base_height: float = 0.2         # maintain target height

    # penalties
    rew_action_smoothness: float = -0.005  # penalise jerk
    rew_energy: float = -0.0001            # penalise torque * velocity
    rew_vel_mismatch: float = 0.5          # penalise z-vel and xy-ang-vel
    rew_alive: float = 0.15               # stay alive bonus
    rew_termination: float = -2.0          # penalty on fall
