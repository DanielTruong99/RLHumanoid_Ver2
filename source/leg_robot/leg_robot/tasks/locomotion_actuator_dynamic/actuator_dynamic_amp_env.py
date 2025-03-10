# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from .actuator_dynamic_amp_env_cfg import ActuatorDynamicEnvCfg
from .motions import MotionLoader


class ActuatorDynamicEnv(DirectRLEnv):
    cfg: ActuatorDynamicEnvCfg

    def __init__(self, cfg: ActuatorDynamicEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        # dof_torque_limits = torch.cat(
        #     (self.robot.actuators["legs"].effort_limit, self.robot.actuators["feet"].effort_limit),
        #     dim=1
        # )
        dof_torque_limits = torch.tensor(self.robot.actuators["legs"].effort_limit, device=self.device)
        self.action_offset = 0.5 * ( 2.0 * 50 )
        self.action_scale = 2.0 * 50

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "is_terminated",
                "alive",
            ]
        }

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device) # type: ignore

        # DOF and key body indexes
        #! Need to be changed
        key_body_names = ["LF_scap", "LF_hip", "LF_knee"]
        jey_joint_names = ["LFJ_scap", "LFJ_hip", "LFJ_knee"]
        self.key_joint_indexes = self.robot.find_joints(jey_joint_names)[0]
        self.ref_body_index = self.robot.data.body_names.index('world')
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(jey_joint_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        # add robot
        self.robot = Articulation(self.cfg.robot)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        """
            The action is the joint torques applied to the robot.
            The action is scaled to the joint torque limits and added to the joint efforts target.
        """
        target = self.action_offset + self.action_scale * self.actions
        # target = self.actions
        self.robot.set_joint_effort_target(target, self.key_joint_indexes)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()

        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos[:, self.key_joint_indexes],
            self.robot.data.joint_vel[:, self.key_joint_indexes],
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # joint torques
        joint_torques = torch.sum(torch.square(self.robot.data.applied_torque[:, self.key_joint_indexes]), dim=1)

        # joint acceleration
        joint_accel = torch.sum(torch.square(self.robot.data.joint_acc[:, self.key_joint_indexes]), dim=1)

        # action rate
        action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)

        rewards = {
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "is_terminated": self.reset_terminated.float() * self.cfg.terminated_scale * self.step_dt,
            "alive": (1.0 - self.reset_terminated.float()) * self.cfg.alive_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            # died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            # check dof position limits
            dof_pos = self.robot.data.joint_pos[:, self.key_joint_indexes]
            dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.key_joint_indexes, 0]
            dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.key_joint_indexes, 1]
            died = torch.any(
                torch.logical_or(
                    dof_pos < dof_lower_limits,
                    dof_pos > dof_upper_limits,
                ),
                dim=-1,
            )
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES # type: ignore
        self.robot.reset(env_ids) # type: ignore
        super()._reset_idx(env_ids) # type: ignore

        if self.cfg.reset_strategy == "default":
            joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self.key_joint_indexes, env_ids) # type: ignore

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            dof_currents,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)


        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            dof_currents,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions,
            dof_velocities,
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
        ),
        dim=-1,
    )
    return obs
