from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat

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
        self._knee_body_ids, _ = self.robot.find_bodies(self.cfg.knee_body_names)
        self._termination_body_ids, _ = self.robot.find_bodies(
            self.cfg.termination_contact_body_names
        )

        # --- contact sensor body index lookups ---
        self._sensor_foot_ids, _ = self._contact_sensor.find_bodies(self.cfg.foot_body_names)
        self._sensor_base_ids, _ = self._contact_sensor.find_bodies(
            self.cfg.termination_contact_body_names
        )

        # --- buffers ---
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)
        self.last_joint_vel = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        # gait phase tracking
        self.sin_phase = torch.zeros(self.num_envs, device=self.device)
        self.cos_phase = torch.zeros(self.num_envs, device=self.device)
        self.ref_joint_pos = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # base velocity tracking (for base_acc reward)
        self.last_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # foot contact tracking
        self.foot_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_foot_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.first_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.air_time_on_contact = torch.zeros(self.num_envs, 2, device=self.device)

        # accumulated foot height during swing (EngineAI-style: reset on contact, accumulate deltas)
        self.feet_heights = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_foot_z = torch.zeros(self.num_envs, 2, device=self.device)

        # track which envs have zero commands (for gait phase freezing)
        self.still_commands = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # command resample counter (in policy steps)
        self._cmd_resample_steps = int(
            self.cfg.cmd_resample_time_s / (self.cfg.sim.dt * self.cfg.decimation)
        )
        self._cmd_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- per-term reward tracking for diagnostics ---
        self._reward_term_names = [
            "tracking_lin_vel", "tracking_ang_vel", "ref_joint_pos", "feet_air_time",
            "contact_pattern", "orientation", "base_height", "vel_mismatch",
            "action_smoothness", "energy", "alive", "feet_clearance",
            "default_joint_pos", "feet_distance", "foot_slip", "track_vel_hard",
            "low_speed", "dof_vel", "dof_acc", "lat_vel", "feet_height_max", "swing_phase_ground",
            "base_acc", "knee_distance", "force_balance", "termination",
        ]
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self._reward_term_names
        }
        # track base velocity for diagnostics
        self._episode_base_vel_x_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)

        # gait phase diagnostics (logged per episode)
        self._episode_foot_height_l_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_foot_height_r_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_foot_force_l_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_foot_force_r_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_swing_count_l = torch.zeros(self.num_envs, device=self.device)
        self._episode_swing_count_r = torch.zeros(self.num_envs, device=self.device)

        # cached tensors
        self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)

        # policy dt for convenience
        self._dt = self.cfg.sim.dt * self.cfg.decimation

        # --- push force (domain randomization) ---
        if self.cfg.push_robots:
            self._push_interval_steps = int(
                self.cfg.push_interval_s / self._dt
            )
            self._push_step_counter = 0

        # --- curriculum: swing penalty annealing ---
        self._global_step_counter = 0

        # --- Run 46: compact history buffer [N, history_len, obs_history_size] ---
        # Stores last 15 frames of: ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18
        # 15 frames matches EngineAI frame_stack=15; compact 18-dim keeps total obs at 334 (< 512 hidden — no bottleneck)
        # Run 44 failed: full 64×15=960 > 512 first hidden layer (compression bottleneck)
        # Run 45 used 3 frames (118-dim) — too short temporal context
        self.obs_history = torch.zeros(
            self.num_envs, self.cfg.obs_history_len, self.cfg.obs_history_size,
            device=self.device
        )

        # --- PD gains randomization ---
        if self.cfg.pd_gains_rand:
            # store original gains (will be populated after scene setup in first reset)
            self._original_stiffness = None
            self._original_damping = None

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

        # contact sensor (force-based contact detection — replaces z-height method)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ================================================================
    # Actions
    # ================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_prev_actions[:] = self.prev_actions
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # store joint velocities for acceleration penalty
        self.last_joint_vel[:] = self.robot.data.joint_vel[:, self._leg_joint_ids]
        # store base velocity for base_acc reward
        self.last_base_lin_vel[:] = self.robot.data.root_lin_vel_b
        # apply random push forces (velocity impulses) at intervals
        self._apply_push_forces()

    def _apply_action(self) -> None:
        # PD position target = default_pos + scale * action
        default_pos = self.robot.data.default_joint_pos[:, self._leg_joint_ids]
        targets = default_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._leg_joint_ids)

    # ================================================================
    # Observations  [N, 334]  (Run 46: 64 current + 270 compact history — 15×18)
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

        # current single-frame observation [N, 64]
        current_obs = torch.cat(
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
        )  # [N, 64]

        # Run 46: compact history — disturbance-relevant signals only [N, 18]
        # ang_vel_b(3) + projected_gravity(3) + joint_pos_rel(12) = 18 dims × 15 frames
        history_frame = torch.cat([ang_vel_b, projected_gravity, joint_pos_rel], dim=-1)  # [N, 18]

        # roll history buffer: shift old frames back, insert current at front
        # obs_history: [N, 15, 18] — index 0 = most recent
        self.obs_history = torch.roll(self.obs_history, shifts=1, dims=1)
        self.obs_history[:, 0, :] = history_frame

        # concatenate current full obs + flattened compact history → [N, 334]
        obs_stacked = torch.cat([current_obs, self.obs_history.view(self.num_envs, -1)], dim=-1)
        return {"policy": obs_stacked}

    # ================================================================
    # Rewards
    # ================================================================

    def _get_rewards(self) -> torch.Tensor:
        # increment global step counter for curriculum
        self._global_step_counter += 1

        # compute current swing penalty from curriculum (only if swing reward is enabled)
        if self.cfg.rew_swing_phase_ground != 0.0:
            progress = min(1.0, self._global_step_counter / self.cfg.swing_curriculum_steps)
            current_swing_penalty = (
                self.cfg.swing_penalty_start
                + progress * (self.cfg.swing_penalty_end - self.cfg.swing_penalty_start)
            )
        else:
            current_swing_penalty = 0.0

        # foot body state: positions [N, 2, 3] and velocities [N, 2, 6]
        foot_pos_w = self.robot.data.body_pos_w[:, self._foot_body_ids, :]
        foot_vel_w = self.robot.data.body_vel_w[:, self._foot_body_ids, :]

        # knee body positions for knee_distance reward
        knee_pos_w = self.robot.data.body_pos_w[:, self._knee_body_ids, :]

        # foot contact forces for force-balance reward
        net_forces = self._contact_sensor.data.net_forces_w
        foot_forces_z = net_forces[:, self._sensor_foot_ids, 2]  # [N, 2] vertical force

        total, reward_terms = compute_rewards(
            cfg=self.cfg,
            dt=self._dt,
            lin_vel_b=self.robot.data.root_lin_vel_b,
            ang_vel_b=self.robot.data.root_ang_vel_b,
            base_quat=self.robot.data.root_quat_w,
            base_pos_z=self.robot.data.root_pos_w[:, 2],
            joint_pos_rel=self.robot.data.joint_pos[:, self._leg_joint_ids]
            - self.robot.data.default_joint_pos[:, self._leg_joint_ids],
            joint_vel=self.robot.data.joint_vel[:, self._leg_joint_ids],
            last_joint_vel=self.last_joint_vel,
            last_base_lin_vel=self.last_base_lin_vel,
            actions=self.actions,
            prev_actions=self.prev_actions,
            prev_prev_actions=self.prev_prev_actions,
            commands=self.commands,
            ref_joint_pos=self.ref_joint_pos,
            sin_phase=self.sin_phase,
            foot_contact=self.last_foot_contact,
            first_contact=self.first_contact,
            air_time_on_contact=self.air_time_on_contact,
            foot_pos_w=foot_pos_w,
            foot_vel_w=foot_vel_w,
            knee_pos_w=knee_pos_w,
            feet_heights=self.feet_heights,
            foot_forces_z=foot_forces_z,
            reset_terminated=self.reset_terminated,
            device=self.device,
            num_envs=self.num_envs,
            current_swing_penalty=current_swing_penalty,
        )

        # accumulate per-term episode sums for diagnostics
        for name, value in reward_terms.items():
            self._episode_reward_sums[name] += value
        self._episode_base_vel_x_sum += self.robot.data.root_lin_vel_b[:, 0]
        self._episode_step_count += 1

        # accumulate gait phase diagnostics
        foot_z = self.robot.data.body_pos_w[:, self._foot_body_ids, 2]  # [N, 2]
        net_forces = self._contact_sensor.data.net_forces_w
        foot_fz = net_forces[:, self._sensor_foot_ids, 2]  # [N, 2] vertical force
        is_swing_l = (self.sin_phase < 0).float()   # left in swing
        is_swing_r = (self.sin_phase >= 0).float()   # right in swing
        self._episode_foot_height_l_sum += foot_z[:, 0] * is_swing_l
        self._episode_foot_height_r_sum += foot_z[:, 1] * is_swing_r
        self._episode_foot_force_l_sum += foot_fz[:, 0]
        self._episode_foot_force_r_sum += foot_fz[:, 1]
        self._episode_swing_count_l += is_swing_l
        self._episode_swing_count_r += is_swing_r

        return total

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

        # --- body contact: forbidden bodies touching ground (force-based) ---
        net_forces = self._contact_sensor.data.net_forces_w  # [N, num_bodies, 3]
        base_forces = torch.norm(net_forces[:, self._sensor_base_ids, :], dim=-1)  # [N, B]
        bad_contact = torch.any(base_forces > self.cfg.contact_force_threshold, dim=-1)

        terminated = fell | bad_contact
        return terminated, time_out

    # ================================================================
    # Reset
    # ================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # --- log per-term reward diagnostics on episode end ---
        if len(env_ids) > 0:
            extras_log = {}
            steps = self._episode_step_count[env_ids].clamp(min=1)
            for name in self._reward_term_names:
                avg = torch.mean(self._episode_reward_sums[name][env_ids])
                extras_log["Episode_Reward/" + name] = avg.item()
            # log mean base velocity (key diagnostic: is the robot actually moving?)
            mean_vel_x = torch.mean(self._episode_base_vel_x_sum[env_ids] / steps)
            extras_log["Episode/mean_base_vel_x"] = mean_vel_x.item()

            # gait phase diagnostics
            swing_l = self._episode_swing_count_l[env_ids].clamp(min=1)
            swing_r = self._episode_swing_count_r[env_ids].clamp(min=1)
            # mean foot height during swing phase [m] — how high feet lift
            extras_log["Episode/swing_foot_height_l"] = torch.mean(
                self._episode_foot_height_l_sum[env_ids] / swing_l
            ).item()
            extras_log["Episode/swing_foot_height_r"] = torch.mean(
                self._episode_foot_height_r_sum[env_ids] / swing_r
            ).item()
            # mean foot force [N] — overall average (stance + swing)
            extras_log["Episode/mean_foot_force_l"] = torch.mean(
                self._episode_foot_force_l_sum[env_ids] / steps
            ).item()
            extras_log["Episode/mean_foot_force_r"] = torch.mean(
                self._episode_foot_force_r_sum[env_ids] / steps
            ).item()
            # swing ratio — fraction of episode spent in swing per foot (~0.5 ideal)
            extras_log["Episode/swing_ratio_l"] = torch.mean(
                self._episode_swing_count_l[env_ids] / steps
            ).item()
            extras_log["Episode/swing_ratio_r"] = torch.mean(
                self._episode_swing_count_r[env_ids] / steps
            ).item()

            self.extras["log"] = extras_log

            # reset episode accumulators
            for name in self._reward_term_names:
                self._episode_reward_sums[name][env_ids] = 0.0
            self._episode_base_vel_x_sum[env_ids] = 0.0
            self._episode_step_count[env_ids] = 0.0
            self._episode_foot_height_l_sum[env_ids] = 0.0
            self._episode_foot_height_r_sum[env_ids] = 0.0
            self._episode_foot_force_l_sum[env_ids] = 0.0
            self._episode_foot_force_r_sum[env_ids] = 0.0
            self._episode_swing_count_l[env_ids] = 0.0
            self._episode_swing_count_r[env_ids] = 0.0

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
        self.prev_prev_actions[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.last_base_lin_vel[env_ids] = 0.0
        self.foot_air_time[env_ids] = 0.0
        self.last_foot_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.air_time_on_contact[env_ids] = 0.0
        self.feet_heights[env_ids] = 0.0
        self.last_foot_z[env_ids] = 0.0
        self.still_commands[env_ids] = False
        self._cmd_counter[env_ids] = 0
        # Run 44: reset observation history for terminated envs (no stale frames from prior episode)
        self.obs_history[env_ids] = 0.0

        # sample new velocity commands
        self._resample_commands(env_ids)

        # --- PD gains randomization (EngineAI-style: ±20% per DOF per reset) ---
        if self.cfg.pd_gains_rand:
            self._randomize_pd_gains(env_ids)

        # --- friction randomization (Run 41: randomize ground friction per reset) ---
        if self.cfg.friction_rand:
            self._randomize_friction(env_ids)

    # ================================================================
    # Helpers — gait phase
    # ================================================================

    def _update_gait_phase(self):
        """Compute gait phase and reference joint positions from episode time.

        Run 40: switched to EngineAI's hip_yaw (idx 2/8) gait reference.
        Drives hip_yaw + knee_pitch + ankle_pitch with coupled amplitudes.
        hip_yaw creates natural knee-lifting motion during swing phase.
        (hip_pitch was used in Runs 5-39, creating pendulum swing without knee bend.)
        Phase is frozen when commands are zero (standing still).
        """
        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        phase = (episode_time / self.cfg.cycle_time) % 1.0
        # freeze phase when standing still (matching EngineAI)
        phase[self.still_commands] = 0.0
        self.sin_phase = torch.sin(2.0 * math.pi * phase)
        self.cos_phase = torch.cos(2.0 * math.pi * phase)

        # generate reference joint positions (sinusoidal bipedal gait)
        # drives hip_yaw (idx 2/8), knee_pitch (idx 3/9), ankle_pitch (idx 4/10)
        # hip_yaw for leg rotation creating natural knee lift (matching EngineAI)
        scale = self.cfg.target_joint_pos_scale   # 0.26 rad
        scale2 = 2.0 * scale                      # 0.52 rad (knee bends at 2× hip)
        self.ref_joint_pos.zero_()

        sin_pos = self.sin_phase.clone()
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        # left leg swings when sin_phase < 0 (zero out positive values)
        sin_pos_l[sin_pos_l > 0] = 0
        # hip_yaw_l (idx 2): leg rotation for knee lift
        self.ref_joint_pos[:, 2] = sin_pos_l * scale
        # knee_pitch_l (idx 3): bend at double amplitude
        self.ref_joint_pos[:, 3] = -sin_pos_l * scale2
        # ankle_pitch_l (idx 4): compensate
        self.ref_joint_pos[:, 4] = sin_pos_l * scale

        # right leg swings when sin_phase > 0 (zero out negative values)
        sin_pos_r[sin_pos_r < 0] = 0
        # hip_yaw_r (idx 8): leg rotation for knee lift
        self.ref_joint_pos[:, 8] = -sin_pos_r * scale
        # knee_pitch_r (idx 9)
        self.ref_joint_pos[:, 9] = sin_pos_r * scale2
        # ankle_pitch_r (idx 10)
        self.ref_joint_pos[:, 10] = -sin_pos_r * scale

        # deadband near zero crossing (matching EngineAI)
        self.ref_joint_pos[torch.abs(sin_pos) < 0.05] = 0.0

    # ================================================================
    # Helpers — foot contact
    # ================================================================

    def _compute_foot_contact(self) -> torch.Tensor:
        """Detect foot contact from contact sensor forces. Returns [N, 2] bool.

        Uses the z-component (vertical) of net contact forces from the physics engine.
        This is more reliable than z-height: a shuffling foot at z=0.15m that barely
        touches the ground produces negligible force and correctly registers as "not in contact".
        EngineAI uses the same approach with a 5N threshold.
        """
        net_forces = self._contact_sensor.data.net_forces_w  # [N, num_bodies, 3]
        foot_forces_z = net_forces[:, self._sensor_foot_ids, 2]  # [N, 2] — vertical force
        return foot_forces_z > self.cfg.contact_force_threshold

    def _update_foot_contact(self):
        """Track foot air time and accumulated swing height for gait rewards."""
        contact = self._compute_foot_contact()

        # EngineAI biped contact filtering: contact OR last_contacts OR stance_mask
        # stance_mask ensures feet register as "in contact" during expected stance phase
        # even before force threshold is reached — critical for gait phase coordination
        stance_mask = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        stance_mask[:, 0] = self.sin_phase >= 0  # left stance when sin >= 0
        stance_mask[:, 1] = self.sin_phase < 0   # right stance when sin < 0
        contact_filt = contact | self.last_foot_contact | stance_mask

        # detect first-contact events: foot was in air AND now filtered-contact
        self.first_contact = (self.foot_air_time > 0.0) & contact_filt

        # save air time at the moment of landing (before reset)
        self.air_time_on_contact = self.foot_air_time * self.first_contact.float()

        # increment air time for feet in the air, reset on filtered contact
        self.foot_air_time += self._dt
        self.foot_air_time *= ~contact_filt  # reset to 0 on filtered contact

        # accumulated foot height (EngineAI-style):
        # track how high each foot has risen since last contact
        foot_z = self.robot.data.body_pos_w[:, self._foot_body_ids, 2]  # [N, 2]
        delta_z = foot_z - self.last_foot_z
        self.feet_heights += delta_z
        self.last_foot_z = foot_z.clone()
        self.feet_heights *= ~contact_filt  # reset on filtered contact

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
        """Sample new velocity commands for given envs.

        Run 13: forward-only commands (0.3-1.0 m/s) to eliminate standing-still exploit.
        No zero commands, no small command filter.
        """
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
        # zero commands (standing still) — disabled for Run 13 (was exploited)
        if self.cfg.cmd_still_ratio > 0:
            still_mask = torch.rand(n, device=self.device) < self.cfg.cmd_still_ratio
            self.commands[env_ids[still_mask]] = 0.0
        # track which envs are standing still (for gait phase freezing)
        self.still_commands[env_ids] = (
            torch.norm(self.commands[env_ids], dim=1) < 0.1
        )

    # ================================================================
    # Helpers — push forces (domain randomization)
    # ================================================================

    def _apply_push_forces(self):
        """Apply random velocity impulses to robot base at fixed intervals.

        Matches EngineAI: every push_interval_s seconds, add random linear (xy)
        and angular (rpy) velocity to the robot base. Forces the robot to take
        reactive steps to maintain balance, preventing the standing-still exploit.
        """
        if not self.cfg.push_robots:
            return
        self._push_step_counter += 1
        if self._push_step_counter % self._push_interval_steps != 0:
            return

        # sample random velocity impulses
        vel = self.robot.data.root_vel_w.clone()  # [N, 6] = [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
        # add random linear velocity in xy
        vel[:, 0] += torch.empty(self.num_envs, device=self.device).uniform_(
            -self.cfg.max_push_vel_xy, self.cfg.max_push_vel_xy
        )
        vel[:, 1] += torch.empty(self.num_envs, device=self.device).uniform_(
            -self.cfg.max_push_vel_xy, self.cfg.max_push_vel_xy
        )
        # add random angular velocity in roll/pitch/yaw
        vel[:, 3:6] += torch.empty(self.num_envs, 3, device=self.device).uniform_(
            -self.cfg.max_push_ang_vel, self.cfg.max_push_ang_vel
        )
        # write back to simulation
        self.robot.write_root_velocity_to_sim(vel)

    # ================================================================
    # Helpers — PD gains randomization
    # ================================================================

    def _randomize_pd_gains(self, env_ids):
        """Randomize stiffness/damping per DOF per reset (EngineAI-style ±20%).

        Stores original gains on first call, then applies random multipliers
        for the reset envs only.
        """
        # lazy-init: capture original gains on first call
        if self._original_stiffness is None:
            self._original_stiffness = self.robot.data.joint_stiffness.clone()
            self._original_damping = self.robot.data.joint_damping.clone()

        n = len(env_ids)
        num_joints = self._original_stiffness.shape[1]

        # random multipliers per DOF
        stiff_multi = torch.empty(n, num_joints, device=self.device).uniform_(
            *self.cfg.stiffness_multi_range
        )
        damp_multi = torch.empty(n, num_joints, device=self.device).uniform_(
            *self.cfg.damping_multi_range
        )

        # apply multiplied gains
        new_stiffness = self._original_stiffness[env_ids] * stiff_multi
        new_damping = self._original_damping[env_ids] * damp_multi

        self.robot.write_joint_stiffness_to_sim(new_stiffness, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(new_damping, env_ids=env_ids)

    # ================================================================
    # Helpers — friction randomization
    # ================================================================

    def _randomize_friction(self, env_ids):
        """Randomize ground friction per reset (Run 41: EngineAI-style).

        Applies a random multiplier to both static and dynamic friction
        for the robot's foot contact materials.
        """
        n = len(env_ids)
        friction_multi = torch.empty(n, device=self.device).uniform_(
            *self.cfg.friction_range
        )
        # Modify robot body material properties per env
        # material_properties shape: [N, num_shapes, 3] = [static_friction, dynamic_friction, restitution]
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long) if not isinstance(env_ids, torch.Tensor) else env_ids
        foot_material = self.robot.root_physx_view.get_material_properties()
        for i, env_id in enumerate(env_ids):
            eid = int(env_id)
            foot_material[eid, :, 0] = friction_multi[i].item()  # static friction
            foot_material[eid, :, 1] = friction_multi[i].item()  # dynamic friction
        indices = env_ids_tensor.to(dtype=torch.int32, device="cpu")
        self.robot.root_physx_view.set_material_properties(foot_material, indices)


# =====================================================================
# Reward computation (standalone for clarity)
# =====================================================================

def compute_rewards(
    cfg: ArmrobotleggingEnvCfg,
    dt: float,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    base_quat: torch.Tensor,
    base_pos_z: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    last_joint_vel: torch.Tensor,
    last_base_lin_vel: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_prev_actions: torch.Tensor,
    commands: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    sin_phase: torch.Tensor,
    foot_contact: torch.Tensor,
    first_contact: torch.Tensor,
    air_time_on_contact: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_vel_w: torch.Tensor,
    knee_pos_w: torch.Tensor,
    feet_heights: torch.Tensor,
    foot_forces_z: torch.Tensor,
    reset_terminated: torch.Tensor,
    device: torch.device,
    num_envs: int,
    current_swing_penalty: float = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # --- 1. velocity tracking ---
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - lin_vel_b[:, :2]), dim=-1)
    rew_lin_vel = cfg.rew_tracking_lin_vel * torch.exp(-lin_vel_error / cfg.rew_tracking_sigma)

    ang_vel_error = torch.square(commands[:, 2] - ang_vel_b[:, 2])
    rew_ang_vel = cfg.rew_tracking_ang_vel * torch.exp(-ang_vel_error / cfg.rew_tracking_sigma)

    # --- 2. gait reference tracking (EngineAI formula: exp-of-norm with penalty) ---
    # exp(-2 * ||diff||) - 0.2 * clamp(||diff||, 0, 0.5)
    # Unlike mean-of-exp, this uses the NORM across all joints — less "free reward"
    # when only a few joints deviate, and adds a linear penalty for deviations up to 0.5
    ref_diff = ref_joint_pos - joint_pos_rel
    diff_norm = torch.norm(ref_diff, dim=1)
    rew_ref_pos = cfg.rew_ref_joint_pos * (torch.exp(-2.0 * diff_norm) - 0.2 * diff_norm.clamp(0, 0.5))

    # --- 3. feet air time (EngineAI BIPED formula: clamp, always positive) ---
    # Biped uses clamp(air_time, 0, 0.5) — rewards ANY step, capped at 0.5s
    # Unlike the quadruped formula (air_time - 0.5) which penalizes steps < 0.5s,
    # the biped formula gives positive reward proportional to air time, making it
    # discoverable through exploration. Only fires on first_contact (landing event).
    rew_air_time = cfg.rew_feet_air_time * torch.sum(
        air_time_on_contact.clamp(0, 0.5) * first_contact.float(), dim=-1
    )
    # Note: EngineAI biped does NOT gate by velocity (quadruped does). No gate here.

    # --- 3b. swing phase ground penalty (continuous signal to force foot lifting) ---
    # When a foot SHOULD be in swing phase but is on the ground, penalize.
    # This provides gradient even when feet never lift (unlike air_time which needs first_contact).
    swing_mask_penalty = torch.zeros(num_envs, 2, device=device)
    swing_mask_penalty[:, 0] = (sin_phase < 0).float()   # left should swing
    swing_mask_penalty[:, 1] = (sin_phase > 0).float()   # right should swing
    # penalty = foot is on ground during swing phase
    swing_ground_violation = swing_mask_penalty * foot_contact.float()  # [N, 2]
    swing_scale = current_swing_penalty if current_swing_penalty is not None else cfg.rew_swing_phase_ground
    rew_swing_ground = swing_scale * torch.sum(swing_ground_violation, dim=-1)
    # only when commanded to move
    rew_swing_ground *= (torch.norm(commands[:, :2], dim=1) > 0.1)

    # --- 4. feet contact pattern (matching EngineAI: +1.0 match, -0.3 mismatch) ---
    # left foot should be in stance when sin_phase >= 0, swing when < 0
    # right foot should be in stance when sin_phase < 0, swing when >= 0
    desired_contact_left = (sin_phase >= 0)
    desired_contact_right = (sin_phase < 0)
    desired_contact = torch.stack([desired_contact_left, desired_contact_right], dim=-1)
    contact_reward = torch.where(foot_contact == desired_contact, 1.0, -0.3)
    rew_contact_pattern = cfg.rew_feet_contact_number * torch.mean(contact_reward, dim=-1)

    # --- 5. orientation (stay upright) --- Run 47: EngineAI dual-signal formula
    # Signal 1: euler roll/pitch angles (scale 10)
    # Signal 2: projected gravity xy norm (scale 20, stronger than our old scale 10)
    # Combined: average of both → reward in [0, 1], no weight rescaling needed
    gravity_w = torch.zeros(num_envs, 3, device=device)
    gravity_w[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(base_quat, gravity_w)
    base_euler = euler_xyz_from_quat(base_quat)  # [N, 3] roll, pitch, yaw
    quat_mismatch = torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=1) * 10.0)
    orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20.0)
    rew_orient = cfg.rew_orientation * (quat_mismatch + orientation) / 2.0

    # --- 6. base height ---
    height_error = torch.abs(base_pos_z - cfg.base_height_target)
    rew_height = cfg.rew_base_height * torch.exp(-height_error * 100.0)

    # --- 7. velocity mismatch (penalise z-vel and xy-ang-vel) ---
    rew_vel_mismatch = cfg.rew_vel_mismatch * 0.5 * (
        torch.exp(-torch.square(lin_vel_b[:, 2]) * 10.0)
        + torch.exp(-torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1) * 5.0)
    )

    # --- 8. action smoothness penalty (EngineAI 2nd-order version) ---
    # term_1: consecutive action difference
    term_1 = torch.sum(torch.square(actions - prev_actions), dim=-1)
    # term_2: 2nd-order smoothness (prevents rapid acceleration of actions)
    term_2 = torch.sum(torch.square(actions + prev_prev_actions - 2.0 * prev_actions), dim=-1)
    # term_3: action magnitude
    term_3 = 0.05 * torch.sum(torch.abs(actions), dim=-1)
    rew_smooth = cfg.rew_action_smoothness * (term_1 + term_2 + term_3)

    # --- 9. energy penalty ---
    rew_energy = cfg.rew_energy * torch.sum(torch.square(actions) * torch.abs(joint_vel), dim=-1)

    # --- 10. alive bonus ---
    rew_alive = cfg.rew_alive * torch.ones(num_envs, device=device)

    # --- 11. feet clearance (swing foot height tracking — EngineAI-style) ---
    # Uses accumulated feet_heights (reset to 0 on contact, tracks rise during swing)
    # instead of absolute z-position which was broken (foot at 0.148m > target 0.10m)
    swing_mask = torch.zeros(num_envs, 2, device=device)
    swing_mask[:, 0] = (sin_phase < 0).float()   # left foot swing
    swing_mask[:, 1] = (sin_phase > 0).float()   # right foot swing
    swing_curve = torch.zeros(num_envs, 2, device=device)
    swing_curve[:, 0] = torch.clamp(-sin_phase, min=0.0)  # left: -sin when sin < 0
    swing_curve[:, 1] = torch.clamp(sin_phase, min=0.0)   # right: sin when sin > 0
    # target height = swing_curve * target_feet_height, penalize deviation
    target_h = swing_curve * cfg.target_feet_height
    clearance_error = (target_h - feet_heights) * swing_mask
    rew_clearance = cfg.rew_feet_clearance * torch.norm(clearance_error, dim=1)

    # --- 11b. max foot height penalty (Run 19: penalize lifting too high) ---
    over_height = torch.clamp(feet_heights - cfg.max_feet_height, min=0.0) * swing_mask
    rew_feet_height_max = cfg.rew_feet_height_max * torch.sum(over_height, dim=-1)

    # --- 12. default joint position --- Run 47: fix indices to target hip splay joints
    # Old code used [0,1,6,7] = hip_pitch_l, hip_roll_l, hip_pitch_r, hip_roll_r (wrong — hip_pitch needs freedom)
    # New: [1,2,7,8] = hip_roll_l, hip_yaw_l, hip_roll_r, hip_yaw_r (the actual splay joints)
    # Formula: EngineAI-style norm + 0.1 rad deadband + mild linear penalty on all joints
    left_splay = joint_pos_rel[:, [1, 2]]   # hip_roll_l, hip_yaw_l
    right_splay = joint_pos_rel[:, [7, 8]]  # hip_roll_r, hip_yaw_r
    splay_norm = torch.norm(left_splay, dim=1) + torch.norm(right_splay, dim=1)
    splay_norm = torch.clamp(splay_norm - 0.1, min=0.0, max=0.5)  # 0.1 rad deadband
    rew_default_pos = cfg.rew_default_joint_pos * (
        torch.exp(-splay_norm * 100.0) - 0.01 * torch.norm(joint_pos_rel, dim=1)
    )

    # --- 13. feet distance (keep feet within proper range) ---
    foot_xy = foot_pos_w[:, :, :2]  # [N, 2, 2] — xy positions of both feet
    foot_dist = torch.norm(foot_xy[:, 0, :] - foot_xy[:, 1, :], dim=1)
    d_min = torch.clamp(foot_dist - cfg.min_feet_dist, max=0.0)   # negative if too close
    d_max = torch.clamp(foot_dist - cfg.max_feet_dist, min=0.0)   # positive if too far
    rew_feet_dist = cfg.rew_feet_distance * 0.5 * (
        torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)
    )

    # --- 14. foot slip penalty (penalize foot velocity during contact) ---
    foot_speed_xy = torch.norm(foot_vel_w[:, :, :2], dim=-1)  # [N, 2] linear vel xy
    foot_slip = torch.sqrt(foot_speed_xy + 1e-6) * foot_contact.float()
    rew_foot_slip = cfg.rew_foot_slip * torch.sum(foot_slip, dim=-1)

    # --- 15. track_vel_hard (sharp velocity tracking — forces actual locomotion) ---
    lin_vel_error_hard = torch.norm(commands[:, :2] - lin_vel_b[:, :2], dim=1)
    lin_vel_error_hard_exp = torch.exp(-lin_vel_error_hard * 10.0)
    ang_vel_error_hard = torch.abs(commands[:, 2] - ang_vel_b[:, 2])
    ang_vel_error_hard_exp = torch.exp(-ang_vel_error_hard * 10.0)
    linear_error_penalty = 0.2 * (lin_vel_error_hard + ang_vel_error_hard)
    rew_track_vel_hard = cfg.rew_track_vel_hard * (
        (lin_vel_error_hard_exp + ang_vel_error_hard_exp) / 2.0 - linear_error_penalty
    )

    # --- 16. low_speed (discrete reward — punish too slow, reward good speed) ---
    abs_speed = torch.abs(lin_vel_b[:, 0])
    abs_command = torch.abs(commands[:, 0])
    speed_too_low = abs_speed < 0.5 * abs_command
    speed_too_high = abs_speed > 1.2 * abs_command
    speed_desired = ~(speed_too_low | speed_too_high)
    sign_mismatch = torch.sign(lin_vel_b[:, 0]) != torch.sign(commands[:, 0])
    rew_low_speed_val = torch.zeros(num_envs, device=device)
    rew_low_speed_val[speed_too_low] = -1.0
    rew_low_speed_val[speed_desired] = 2.0
    rew_low_speed_val[sign_mismatch] = -2.0   # highest priority
    rew_low_speed_val *= (abs_command > 0.1)   # only when commanded to move
    rew_low_speed = cfg.rew_low_speed * rew_low_speed_val

    # --- 17. dof_vel penalty (penalize all joint velocities — discourages vibration) ---
    rew_dof_vel = cfg.rew_dof_vel * torch.sum(torch.square(joint_vel), dim=-1)

    # --- 18. dof_acc penalty (penalize joint accelerations — CRITICAL for anti-vibration) ---
    dof_acc = (joint_vel - last_joint_vel) / dt
    rew_dof_acc = cfg.rew_dof_acc * torch.sum(torch.square(dof_acc), dim=-1)

    # --- 19. knee_distance (prevent sideways shuffling — keep knees properly spaced) ---
    # use knee body positions — approximate from foot positions and base
    # knee bodies: link_knee_pitch_l, link_knee_pitch_r
    # For now, use the XY distance between feet as proxy (already have foot_pos_w)
    # We add a specific reward for lateral (Y) velocity tracking
    lat_vel_error = torch.square(commands[:, 1] - lin_vel_b[:, 1])
    rew_lat_vel = cfg.rew_lat_vel * torch.exp(-lat_vel_error * 10.0)

    # --- 20. base acceleration reward (Run 39: EngineAI anti-wobble) ---
    # Rewards smooth base motion: exp(-norm(base_acc) * 3)
    base_acc = (lin_vel_b - last_base_lin_vel) / dt
    base_acc_norm = torch.norm(base_acc, dim=1)
    rew_base_acc = cfg.rew_base_acc * torch.exp(-base_acc_norm * 3.0)

    # --- 21. knee distance reward (Run 39: EngineAI anti-wobble) ---
    # Penalize knees being too close (prevents collision) or too far (unnatural gait)
    knee_xy = knee_pos_w[:, :, :2]  # [N, 2, 2]
    knee_dist = torch.norm(knee_xy[:, 0, :] - knee_xy[:, 1, :], dim=1)
    knee_dist_error = torch.abs(knee_dist - cfg.target_knee_dist)
    rew_knee_dist = cfg.rew_knee_distance * torch.exp(-knee_dist_error * 20.0)

    # --- 22. force balance reward (Run 41: address L/R asymmetry) ---
    # Reward equal left/right ground reaction forces.
    # force_ratio = min(L,R) / (L+R) → 0.5 when perfectly balanced, 0 when one-sided.
    force_l = foot_forces_z[:, 0].clamp(min=0.0)  # [N] left foot vertical force
    force_r = foot_forces_z[:, 1].clamp(min=0.0)  # [N] right foot vertical force
    force_sum = force_l + force_r + 1e-6  # avoid div-by-zero
    force_ratio = torch.min(force_l, force_r) / force_sum
    # exp reward centered on 0.5 (perfect balance)
    rew_force_balance = cfg.rew_force_balance * torch.exp(-torch.square(force_ratio - 0.5) * 50.0)

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
        + rew_default_pos
        + rew_feet_dist
        + rew_track_vel_hard
        + rew_low_speed
        + rew_lat_vel
        + rew_base_acc
        + rew_knee_dist
        + rew_force_balance
    )
    total_positive = torch.clamp(total_positive, min=0.0)

    total = (total_positive + rew_smooth + rew_energy + rew_foot_slip
             + rew_clearance + rew_feet_height_max + rew_dof_vel + rew_dof_acc + rew_swing_ground)

    rew_term = cfg.rew_termination * reset_terminated.float()
    total += rew_term

    # per-term dict for diagnostics
    reward_terms = {
        "tracking_lin_vel": rew_lin_vel,
        "tracking_ang_vel": rew_ang_vel,
        "ref_joint_pos": rew_ref_pos,
        "feet_air_time": rew_air_time,
        "contact_pattern": rew_contact_pattern,
        "orientation": rew_orient,
        "base_height": rew_height,
        "vel_mismatch": rew_vel_mismatch,
        "action_smoothness": rew_smooth,
        "energy": rew_energy,
        "alive": rew_alive,
        "feet_clearance": rew_clearance,
        "default_joint_pos": rew_default_pos,
        "feet_distance": rew_feet_dist,
        "foot_slip": rew_foot_slip,
        "track_vel_hard": rew_track_vel_hard,
        "low_speed": rew_low_speed,
        "dof_vel": rew_dof_vel,
        "dof_acc": rew_dof_acc,
        "lat_vel": rew_lat_vel,
        "feet_height_max": rew_feet_height_max,
        "swing_phase_ground": rew_swing_ground,
        "base_acc": rew_base_acc,
        "knee_distance": rew_knee_dist,
        "force_balance": rew_force_balance,
        "termination": rew_term,
    }

    return total, reward_terms
