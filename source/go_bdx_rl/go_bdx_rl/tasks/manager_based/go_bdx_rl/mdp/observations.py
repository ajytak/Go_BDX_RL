import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.managers.manager_base import ManagerTermBase
import isaaclab.utils.math as math_utils

import torch
import numpy as np

def joint_pos_and_vel(env, asset_cfg):
    """Return joint positions and velocities concatenated along the last dimension.

    Only joints configured in asset_cfg.joint_ids are used.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # shape: (num_envs, num_joints)
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # concat on last dimension
    return torch.cat([pos, vel], dim=-1)

def local_root_pos(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the asset root position in the environment frame (relative to env origin).
    """
    # Get the asset's global position
    root_pos_w = env.scene[asset_cfg.name].data.root_pos_w  # [INDEX 1.3.6]
    # Get the environment origins
    env_origins = env.scene.env_origins
    # Subtract origins to get local position
    local_pos = root_pos_w - env_origins
    return local_pos

def quat_yaw_aligned_data(env, asset_cfg=SceneEntityCfg("robot")):
    """
    Computes the asset root orientation relative to its yaw-aligned frame.
    The result is a quaternion representing the roll and pitch relative to the heading.
    """
    # 1. Get the asset's current orientation in the world frame (w, x, y, z)
    root_quat_w = env.scene[asset_cfg.name].data.root_quat_w
    root_lin_vel_w = env.scene[asset_cfg.name].data.root_lin_vel_w
    root_ang_vel_w = env.scene[asset_cfg.name].data.root_ang_vel_w 


    # 2. Extract the yaw angle from the world quaternion
    yaw_w = math_utils.yaw_quat(root_quat_w)

    # 3. Invert the yaw-only quaternion to get the transformation from world-yaw to world frame
    quat_inv_yaw_w = math_utils.quat_inv(yaw_w)

    # 5. Multiply the inverse yaw quaternion by the actual world quaternion
    quat_yaw_aligned_b = math_utils.quat_mul(quat_inv_yaw_w, root_quat_w)
    root_lin_vel_yaw_aligned = math_utils.quat_apply(quat_inv_yaw_w, root_lin_vel_w)
    root_ang_vel_yaw_aligned = math_utils.quat_apply(quat_inv_yaw_w, root_ang_vel_w)

    return quat_yaw_aligned_b, root_lin_vel_yaw_aligned, root_ang_vel_yaw_aligned, quat_inv_yaw_w

def foot_data_from_bodies(env, robot_cfg: SceneEntityCfg, quat_inv_yaw_w):
    """
    Retrieves and processes position (Z-only, local) and velocity (yaw-aligned) 
    for all foot bodies, and returns a single concatenated tensor.
    """
    asset = env.scene[robot_cfg.name]

    # 1. Get body positions and velocities in world frame
    # Shape: (N, 2, 3) where index 0 is Left Foot and index 1 is Right Foot
    body_pos_w = asset.data.body_pos_w[:,robot_cfg.body_ids,:]
    body_lin_vel_w = asset.data.body_lin_vel_w[:,robot_cfg.body_ids,:]

    # 2. Process Positions (Local Z only)
    env_origins = env.scene.env_origins
    left_foot_pos = body_pos_w[:,0,:]-env_origins
    right_foot_pos = body_pos_w[:,1,:]-env_origins 

    # 3. Process Velocities (Yaw-Aligned)
    left_foot_lin_vel_aligned = math_utils.quat_apply(quat_inv_yaw_w, body_lin_vel_w[:,0,:])
    right_foot_lin_vel_aligned = math_utils.quat_apply(quat_inv_yaw_w, body_lin_vel_w[:,1,:])

    foot_obs = torch.cat([
        left_foot_pos[:,2:3], 
        right_foot_pos[:,2:3], 
        left_foot_lin_vel_aligned, 
        right_foot_lin_vel_aligned, 
        ], dim=-1) # Final shape: (N, 8)
    return foot_obs

def amp_observation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Full AMP observation vector."""

    # root position (local)
    root_pos = local_root_pos(env, asset_cfg)             # (N, 3)

    # yaw-aligned orientation
    root_quat, root_lin_vel, root_ang_vel, quat_inv_yaw_w = quat_yaw_aligned_data(env, asset_cfg)          # (N, 4)

    # foot pos and vel
    foot_pos_and_vel = foot_data_from_bodies(env, foot_asset_cfg, quat_inv_yaw_w) # (N,8)

    # joints
    joint_pv = joint_pos_and_vel(env, asset_cfg)          # (N, 16*2)
    
    # final stack
    amp_obs = torch.cat([
        root_pos[:,2:3],
        root_quat,
        root_lin_vel,
        root_ang_vel,
        foot_pos_and_vel,
        joint_pv
    ], dim=-1)

    return amp_obs