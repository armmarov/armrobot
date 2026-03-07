from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate_inverse

from .armrobotlegging_env_cfg import ArmrobotleggingEnvCfg


class ArmrobotleggingEnv(DirectRLEnv):
    """PM01 bipedal walking environment using Direct RL workflow.

    Follows the gait-phase reference approach from the EngineAI PM01 walking task,
    ported to the IsaacLab DirectRLEnv API.
    """

    cfg: ArmrobotleggingEnvCfg

    def __init__(self, cfg: ArmrobotleggingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- joint / body index lookups ---
        self._leg_joint_ids, _ = self.robot.find_joints(self.cfg.leg_joint_names)
        self._foot_body_ids, _ = self.robot.find_bodies(self.cfg.foot_body_names)
        self._termination_body_ids, _ = self.robot.find_bodies(
            self.cfg.termination_contact_body_names
        )

        # --- buffers ---
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        # gait phase tracking
        self.sin_phase = torch.zeros(self.num_envs, device=self.device)
        self.cos_phase = torch.zeros(self.num_envs, device=self.device)
        self.ref_joint_pos = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # foot contact tracking
        self.foot_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_foot_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.first_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.air_time_on_contact = torch.zeros(self.num_envs, 2, device=self.device)

        # command resample counter (in policy steps)
        self._cmd_resample_steps = int(
            self.cfg.cmd_resample_time_s / (self.cfg.sim.dt * self.cfg.decimation)
        )
        self._cmd_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # cached tensors
        self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)

        # policy dt for convenience
        self._dt = self.cfg.sim.dt * self.cfg.decimation

    # ================================================================
    # Scene
    # ================================================================

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self.robot

        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ================================================================
    # Actions
    # ================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # PD position target = default_pos + scale * action
        default_pos = self.robot.data.default_joint_pos[:, self._leg_joint_ids]
        targets = default_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._leg_joint_ids)

    # ================================================================
    # Observations  [N, 64]
    # ================================================================

    def _get_observations(self) -> dict:
        base_quat = self.robot.data.root_quat_w
        lin_vel_b = self.robot.data.root_lin_vel_b
        ang_vel_b = self.robot.data.root_ang_vel_b
        joint_pos = self.robot.data.joint_pos[:, self._leg_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._leg_joint_ids]
        default_pos = self.robot.data.default_joint_pos[:, self._leg_joint_ids]

        # projected gravity in body frame — (0,0,-1) when upright
        projected_gravity = quat_rotate_inverse(base_quat, self._gravity_w.expand(self.num_envs, 3))

        # joint positions relative to default standing pose
        joint_pos_rel = joint_pos - default_pos

        # reference joint position difference
        ref_diff = self.ref_joint_pos - joint_pos_rel

        # foot contact mask (binary)
        contact_mask = self._compute_foot_contact().float()

        obs = torch.cat(
            [
                lin_vel_b,                                          # [N, 3]
                ang_vel_b,                                          # [N, 3]
                projected_gravity,                                  # [N, 3]
                joint_pos_rel,                                      # [N, 12]
                joint_vel,                                          # [N, 12]
                self.prev_actions,                                  # [N, 12]
                self.commands,                                      # [N, 3]
                self.sin_phase.unsqueeze(-1),                       # [N, 1]
                self.cos_phase.unsqueeze(-1),                       # [N, 1]
                ref_diff,                                           # [N, 12]
                contact_mask,                                       # [N, 2]
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ================================================================
    # Rewards
    # ================================================================

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            cfg=self.cfg,
            lin_vel_b=self.robot.data.root_lin_vel_b,
            ang_vel_b=self.robot.data.root_ang_vel_b,
            base_quat=self.robot.data.root_quat_w,
            base_pos_z=self.robot.data.root_pos_w[:, 2],
            joint_pos_rel=self.robot.data.joint_pos[:, self._leg_joint_ids]
            - self.robot.data.default_joint_pos[:, self._leg_joint_ids],
            joint_vel=self.robot.data.joint_vel[:, self._leg_joint_ids],
            actions=self.actions,
            prev_actions=self.prev_actions,
            commands=self.commands,
            ref_joint_pos=self.ref_joint_pos,
            sin_phase=self.sin_phase,
            foot_contact=self.last_foot_contact,
            first_contact=self.first_contact,
            air_time_on_contact=self.air_time_on_contact,
            reset_terminated=self.reset_terminated,
            device=self.device,
            num_envs=self.num_envs,
        )

    # ================================================================
    # Termination
    # ================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # update gait phase, foot contact, and command counter
        self._update_gait_phase()
        self._update_foot_contact()
        self._update_commands()

        # --- timeout ---
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # --- fell: base too low ---
        fell = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height

        # --- body contact: forbidden bodies touching ground ---
        body_pos = self.robot.data.body_pos_w[:, self._termination_body_ids, :]  # [N, B, 3]
        body_heights = body_pos[:, :, 2]  # [N, B]
        bad_contact = torch.any(body_heights < self.cfg.contact_height_threshold, dim=-1)

        terminated = fell | bad_contact
        return terminated, time_out

    # ================================================================
    # Reset
    # ================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot to default state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.foot_air_time[env_ids] = 0.0
        self.last_foot_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.air_time_on_contact[env_ids] = 0.0
        self._cmd_counter[env_ids] = 0

        # sample new velocity commands
        self._resample_commands(env_ids)

    # ================================================================
    # Helpers — gait phase
    # ================================================================

    def _update_gait_phase(self):
        """Compute gait phase and reference joint positions from episode time."""
        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        phase = (episode_time / self.cfg.cycle_time) % 1.0
        self.sin_phase = torch.sin(2.0 * math.pi * phase)
        self.cos_phase = torch.cos(2.0 * math.pi * phase)

        # generate reference joint positions (sinusoidal bipedal gait)
        scale = self.cfg.target_joint_pos_scale
        self.ref_joint_pos.zero_()

        # left leg swings when sin_phase < 0
        left_swing = self.sin_phase < 0
        # hip_pitch_l (idx 0): flex forward
        self.ref_joint_pos[:, 0] = torch.where(
            left_swing, self.sin_phase * scale, self.ref_joint_pos[:, 0]
        )
        # knee_pitch_l (idx 3): bend
        self.ref_joint_pos[:, 3] = torch.where(
            left_swing, -self.sin_phase * 2.0 * scale, self.ref_joint_pos[:, 3]
        )
        # ankle_pitch_l (idx 4): compensate
        self.ref_joint_pos[:, 4] = torch.where(
            left_swing, self.sin_phase * scale, self.ref_joint_pos[:, 4]
        )

        # right leg swings when sin_phase > 0
        right_swing = self.sin_phase > 0
        # hip_pitch_r (idx 6)
        self.ref_joint_pos[:, 6] = torch.where(
            right_swing, -self.sin_phase * scale, self.ref_joint_pos[:, 6]
        )
        # knee_pitch_r (idx 9)
        self.ref_joint_pos[:, 9] = torch.where(
            right_swing, self.sin_phase * 2.0 * scale, self.ref_joint_pos[:, 9]
        )
        # ankle_pitch_r (idx 10)
        self.ref_joint_pos[:, 10] = torch.where(
            right_swing, -self.sin_phase * scale, self.ref_joint_pos[:, 10]
        )

    # ================================================================
    # Helpers — foot contact
    # ================================================================

    def _compute_foot_contact(self) -> torch.Tensor:
        """Estimate foot contact from foot body z-position. Returns [N, 2] bool."""
        foot_pos = self.robot.data.body_pos_w[:, self._foot_body_ids, :]  # [N, 2, 3]
        foot_heights = foot_pos[:, :, 2]  # [N, 2]
        return foot_heights < self.cfg.contact_height_threshold

    def _update_foot_contact(self):
        """Track foot air time for gait rewards."""
        contact = self._compute_foot_contact()
        # detect first-contact events (foot just landed)
        self.first_contact = contact & ~self.last_foot_contact
        # save air time at the moment of landing (before reset)
        self.air_time_on_contact = self.foot_air_time * self.first_contact.float()
        # increment air time for feet in the air, reset on contact
        self.foot_air_time += self._dt
        self.foot_air_time *= ~contact  # reset to 0 on contact
        self.last_foot_contact = contact

    # ================================================================
    # Helpers — velocity commands
    # ================================================================

    def _update_commands(self):
        """Resample velocity commands periodically."""
        self._cmd_counter += 1
        resample_mask = self._cmd_counter >= self._cmd_resample_steps
        if resample_mask.any():
            env_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
            self._resample_commands(env_ids)
            self._cmd_counter[env_ids] = 0

    def _resample_commands(self, env_ids):
        """Sample new velocity commands for given envs."""
        n = len(env_ids)
        self.commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.cmd_lin_vel_x_range
        )
        self.commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.cmd_lin_vel_y_range
        )
        self.commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.cmd_ang_vel_z_range
        )
        # some envs get zero commands (standing still)
        still_mask = torch.rand(n, device=self.device) < self.cfg.cmd_still_ratio
        self.commands[env_ids[still_mask]] = 0.0


# =====================================================================
# Reward computation (standalone for clarity)
# =====================================================================

def compute_rewards(
    cfg: ArmrobotleggingEnvCfg,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    base_quat: torch.Tensor,
    base_pos_z: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    commands: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    sin_phase: torch.Tensor,
    foot_contact: torch.Tensor,
    first_contact: torch.Tensor,
    air_time_on_contact: torch.Tensor,
    reset_terminated: torch.Tensor,
    device: torch.device,
    num_envs: int,
) -> torch.Tensor:

    # --- 1. velocity tracking ---
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - lin_vel_b[:, :2]), dim=-1)
    rew_lin_vel = cfg.rew_tracking_lin_vel * torch.exp(-lin_vel_error / cfg.rew_tracking_sigma)

    ang_vel_error = torch.square(commands[:, 2] - ang_vel_b[:, 2])
    rew_ang_vel = cfg.rew_tracking_ang_vel * torch.exp(-ang_vel_error / cfg.rew_tracking_sigma)

    # --- 2. gait reference tracking ---
    ref_diff_sq = torch.sum(torch.square(ref_joint_pos - joint_pos_rel), dim=-1)
    rew_ref_pos = cfg.rew_ref_joint_pos * torch.exp(-2.0 * ref_diff_sq)

    # --- 3. feet air time ---
    # reward when foot lands (first contact) proportional to how long it was in the air
    # air_time_on_contact was captured BEFORE reset, so it has the actual swing duration
    rew_air_time = cfg.rew_feet_air_time * torch.sum(
        (air_time_on_contact - 0.5) * first_contact.float(), dim=-1
    )

    # --- 4. feet contact pattern ---
    # left foot should be in stance when sin_phase >= 0, swing when < 0
    # right foot should be in stance when sin_phase < 0, swing when >= 0
    contact_f = foot_contact.float()
    desired_contact_left = (sin_phase >= 0).float()
    desired_contact_right = (sin_phase < 0).float()
    desired_contact = torch.stack([desired_contact_left, desired_contact_right], dim=-1)
    contact_match = 1.0 - torch.abs(desired_contact - contact_f)
    rew_contact_pattern = cfg.rew_feet_contact_number * torch.mean(contact_match, dim=-1)

    # --- 5. orientation (stay upright) ---
    gravity_w = torch.zeros(num_envs, 3, device=device)
    gravity_w[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(base_quat, gravity_w)
    # projected_gravity[:, 2] should be -1 when upright
    roll_pitch_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    rew_orient = cfg.rew_orientation * torch.exp(-roll_pitch_error * 10.0)

    # --- 6. base height ---
    height_error = torch.abs(base_pos_z - cfg.base_height_target)
    rew_height = cfg.rew_base_height * torch.exp(-height_error * 100.0)

    # --- 7. velocity mismatch (penalise z-vel and xy-ang-vel) ---
    rew_vel_mismatch = cfg.rew_vel_mismatch * 0.5 * (
        torch.exp(-torch.square(lin_vel_b[:, 2]) * 10.0)
        + torch.exp(-torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1) * 5.0)
    )

    # --- 8. action smoothness penalty ---
    action_diff = torch.sum(torch.square(actions - prev_actions), dim=-1)
    action_mag = torch.sum(torch.square(actions), dim=-1)
    rew_smooth = cfg.rew_action_smoothness * (action_diff + 0.5 * action_mag)

    # --- 9. energy penalty ---
    rew_energy = cfg.rew_energy * torch.sum(torch.square(actions) * torch.abs(joint_vel), dim=-1)

    # --- 10. alive bonus ---
    rew_alive = cfg.rew_alive * torch.ones(num_envs, device=device)

    # --- sum all positive rewards, clamp >= 0, then add penalties ---
    total_positive = (
        rew_lin_vel
        + rew_ang_vel
        + rew_ref_pos
        + rew_air_time
        + rew_contact_pattern
        + rew_orient
        + rew_height
        + rew_vel_mismatch
        + rew_alive
    )
    total_positive = torch.clamp(total_positive, min=0.0)

    total = total_positive + rew_smooth + rew_energy

    # termination penalty (applied after clamping — always felt)
    total += cfg.rew_termination * reset_terminated.float()

    return total
