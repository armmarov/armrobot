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
    # env
    decimation = 4                  # policy runs at 200/4 = 50 Hz
    episode_length_s = 20.0

    # spaces
    # actions  : 12 leg joints (position targets offset from default)
    # obs      : base_height(1) + lin_vel_b(3) + ang_vel_b(3) +
    #            euler_xyz(3) + joint_pos(12) + joint_vel(12) +
    #            commands(3) + prev_actions(12) = 49
    action_space = 12
    observation_space = 49
    state_space = 0

    # simulation  — 200 Hz physics
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # flat ground plane with friction
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = PM01_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # leg joint names in the order the policy controls them
    leg_joint_names: list = [
        "j00_hip_pitch_l",
        "j01_hip_roll_l",
        "j02_hip_yaw_l",
        "j03_knee_pitch_l",
        "j04_ankle_pitch_l",
        "j05_ankle_roll_l",
        "j06_hip_pitch_r",
        "j07_hip_roll_r",
        "j08_hip_yaw_r",
        "j09_knee_pitch_r",
        "j10_ankle_pitch_r",
        "j11_ankle_roll_r",
    ]

    # action scale — policy output is multiplied by this before adding to default pos
    action_scale: float = 0.5  # [rad]

    # velocity command ranges
    cmd_lin_vel_x_range: tuple = (-1.0, 1.0)   # [m/s]
    cmd_lin_vel_y_range: tuple = (-0.5, 0.5)   # [m/s]
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)   # [rad/s]

    # termination
    termination_height: float = 0.4    # reset if base drops below this [m]

    # reward scales
    rew_lin_vel_xy: float = 1.5        # track forward/lateral velocity commands
    rew_ang_vel_z: float = 0.75        # track yaw rate command
    rew_alive: float = 0.5             # stay alive bonus
    rew_action_rate: float = -0.01     # penalise large action changes
    rew_energy: float = -0.0002        # penalise joint torque * velocity
    rew_upright: float = 0.2           # reward for staying upright
    rew_termination: float = -2.0      # penalty on fall
