from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

from .armrobotlegging_env_cfg import ArmrobotleggingEnvCfg


class ArmrobotleggingEnv(DirectRLEnv):
    cfg: ArmrobotleggingEnvCfg

    def __init__(self, cfg: ArmrobotleggingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # resolve leg joint indices from the articulation
        self._leg_joint_ids, _ = self.robot.find_joints(self.cfg.leg_joint_names)

        # velocity commands buffer [num_envs, 3] — (vx, vy, yaw_rate)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        # previous actions for smoothness penalty
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone all envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action pipeline  (called every policy step)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # PD position control: target = default_pos + scale * action
        default_pos = self.robot.data.default_joint_pos[:, self._leg_joint_ids]
        targets = default_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._leg_joint_ids)

    # ------------------------------------------------------------------
    # Observations  [num_envs, 49]
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        base_pos = self.robot.data.root_pos_w                  # [N, 3]
        base_quat = self.robot.data.root_quat_w                # [N, 4]
        lin_vel_b = self.robot.data.root_lin_vel_b             # [N, 3]  body frame
        ang_vel_b = self.robot.data.root_ang_vel_b             # [N, 3]  body frame
        joint_pos = self.robot.data.joint_pos[:, self._leg_joint_ids]  # [N, 12]
        joint_vel = self.robot.data.joint_vel[:, self._leg_joint_ids]  # [N, 12]

        roll, pitch, yaw = euler_xyz_from_quat(base_quat)     # each [N]
        euler_xyz = torch.stack([roll, pitch, yaw], dim=-1)    # [N, 3]

        # joint pos relative to default standing pose
        default_pos = self.robot.data.default_joint_pos[:, self._leg_joint_ids]
        joint_pos_rel = joint_pos - default_pos

        obs = torch.cat(
            [
                base_pos[:, 2:3],      # base height          [N, 1]
                lin_vel_b,             # body-frame lin vel   [N, 3]
                ang_vel_b,             # body-frame ang vel   [N, 3]
                euler_xyz,             # roll, pitch, yaw     [N, 3]
                joint_pos_rel,         # leg joint positions  [N, 12]
                joint_vel,             # leg joint velocities [N, 12]
                self.commands,         # velocity commands    [N, 3]
                self.prev_actions,     # previous actions     [N, 12]
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_b = self.robot.data.root_lin_vel_b   # [N, 3]
        ang_vel_b = self.robot.data.root_ang_vel_b   # [N, 3]
        base_quat = self.robot.data.root_quat_w      # [N, 4]
        joint_vel = self.robot.data.joint_vel[:, self._leg_joint_ids]

        # 1. track commanded linear velocity (xy)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - lin_vel_b[:, :2]), dim=-1
        )
        rew_lin_vel = self.cfg.rew_lin_vel_xy * torch.exp(-lin_vel_error / 0.25)

        # 2. track commanded yaw rate
        ang_vel_error = torch.square(self.commands[:, 2] - ang_vel_b[:, 2])
        rew_ang_vel = self.cfg.rew_ang_vel_z * torch.exp(-ang_vel_error / 0.25)

        # 3. alive bonus
        rew_alive = self.cfg.rew_alive * torch.ones(self.num_envs, device=self.device)

        # 4. action rate penalty
        rew_action_rate = self.cfg.rew_action_rate * torch.sum(
            torch.square(self.actions - self.prev_actions), dim=-1
        )

        # 5. energy penalty  (torque ≈ stiffness * pos_error, but we approximate via joint_vel * actions)
        rew_energy = self.cfg.rew_energy * torch.sum(
            torch.square(self.actions) * torch.abs(joint_vel), dim=-1
        )

        # 6. upright reward — z-axis of base should point up
        # up_vec in world frame for each robot: rotate [0,0,1] by quat
        up_world = torch.zeros(self.num_envs, 3, device=self.device)
        up_world[:, 2] = 1.0
        gravity_b = quat_rotate_inverse(base_quat, up_world)
        rew_upright = self.cfg.rew_upright * gravity_b[:, 2]  # 1.0 when upright

        # 7. termination penalty
        rew_termination = self.cfg.rew_termination * self.reset_terminated.float()

        total = (
            rew_lin_vel
            + rew_ang_vel
            + rew_alive
            + rew_action_rate
            + rew_energy
            + rew_upright
            + rew_termination
        )

        # update previous actions
        self.prev_actions[:] = self.actions

        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self.robot.data.root_pos_w[:, 2]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = base_height < self.cfg.termination_height
        return fell, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot state to default
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # resample velocity commands for reset envs
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.cmd_lin_vel_x_range
        )
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.cmd_lin_vel_y_range
        )
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.cmd_ang_vel_z_range
        )

        # reset previous actions
        self.prev_actions[env_ids] = 0.0
