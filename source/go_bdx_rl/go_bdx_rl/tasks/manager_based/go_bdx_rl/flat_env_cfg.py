# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import  LatentBDXRoughEnvCfg ,BDXRoughEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from go_bdx_rl.tasks.manager_based.go_bdx_rl.motions.motion_loader import MotionLoader
import numpy as np
import torch
import gymnasium as gym



@configclass
class BDXFlatEnvCfg(BDXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6

class BDXFlatEnvCfg_PLAY(BDXFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (1.57, 1.57)

@configclass
class BDXFlatAMPEnvCfg(BDXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.amp.motion_file = "/workspace/isaac_projects/go_bdx_rl/source/go_bdx_rl/go_bdx_rl/robot/go_bdx/custom_jump_solved.npz"
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no commands
        self.commands = None
        self.observations.policy.velocity_commands = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        # self.terminations.base_orientation = None
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.events.push_robot = None
        self.events.base_external_force_torque = None

@configclass
class LatentBDXFlatAMPEnvCfg(LatentBDXRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.amp.motion_file = "/workspace/isaac_projects/go_bdx_rl/source/go_bdx_rl/go_bdx_rl/robot/go_bdx/custom_jump_solved.npz"
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        # self.terminations.base_orientation = None
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        


class BDXFlatAMPEnv(ManagerBasedRLEnv):
    cfg: BDXFlatAMPEnvCfg

    def __init__(self, cfg: BDXFlatAMPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_loader = MotionLoader(npz_file=self.cfg.amp.motion_file, device=self.device)
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.amp.num_amp_observations*self.cfg.amp.amp_observation_space,))

    def step(self, action):
        # ---- call original step
        obs, rew, terminated, truncated, infos = super().step(action)

        # ---- compute AMP obs here
        amp_obs = self.observation_manager.compute_group("amp_obs")
        infos["amp_obs"] = amp_obs

        return obs, rew, terminated, truncated, infos

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.amp.num_amp_observations)
        ).flatten()
        # get motions
        (   root_pos,
            root_quat,
            root_lin_vel_w,
            root_ang_vel_w,
            left_foot_pos,
            right_foot_pos,
            left_foot_vel,
            right_foot_vel,
            dof_positions,
            dof_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = torch.cat((root_pos[...,2:3],
                                     root_quat,
                                     root_lin_vel_w,
                                     root_ang_vel_w,
                                     left_foot_pos[...,2:3],
                                     right_foot_pos[...,2:3],
                                     left_foot_vel,
                                     right_foot_vel,
                                     dof_positions,
                                     dof_velocities), dim=-1)
        return amp_observation.view(-1, self.cfg.amp.num_amp_observations*self.cfg.amp.amp_observation_space)

class LatentBDXFlatAMPEnv(ManagerBasedRLEnv):
    cfg: LatentBDXFlatAMPEnvCfg

    def __init__(self, cfg: LatentBDXFlatAMPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_loader = MotionLoader(npz_file=self.cfg.amp.motion_file, device=self.device)
        #add latent dimension
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2+(self.cfg.amp.num_amp_observations*self.cfg.amp.amp_observation_space),))

    def step(self, action):
        # ---- call original step
        obs, rew, terminated, truncated, infos = super().step(action)

        # ---- compute AMP obs here
        amp_obs = self.observation_manager.compute_group("amp_obs")
        infos["amp_obs"] = amp_obs

        return obs, rew, terminated, truncated, infos

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.amp.num_amp_observations)
        ).flatten()
        # get motions
        (   root_pos,
            root_quat,
            root_lin_vel_w,
            root_ang_vel_w,
            left_foot_pos,
            right_foot_pos,
            left_foot_vel,
            right_foot_vel,
            dof_positions,
            dof_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = torch.cat((root_pos[...,2:3],
                                     root_quat,
                                     root_lin_vel_w,
                                     root_ang_vel_w,
                                     left_foot_pos[...,2:3],
                                     right_foot_pos[...,2:3],
                                     left_foot_vel,
                                     right_foot_vel,
                                     dof_positions,
                                     dof_velocities), dim=-1)
        return amp_observation.view(-1, self.cfg.amp.num_amp_observations*self.cfg.amp.amp_observation_space)