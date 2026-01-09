"""
Custom observation functions for robot arm manipulation tasks.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def ee_position(env: ManagerBasedRLEnv, asset_cfg: str = "robot") -> torch.Tensor:
    """
    End-effector position in world frame.
    
    Args:
        env: The RL environment.
        asset_cfg: Name of the robot asset in the scene.
        
    Returns:
        End-effector position tensor of shape (num_envs, 3).
    """
    robot = env.scene[asset_cfg]
    return robot.data.body_pos_w[:, -1, :]


def ee_orientation(env: ManagerBasedRLEnv, asset_cfg: str = "robot") -> torch.Tensor:
    """
    End-effector orientation quaternion in world frame.
    
    Args:
        env: The RL environment.
        asset_cfg: Name of the robot asset in the scene.
        
    Returns:
        End-effector orientation quaternion tensor of shape (num_envs, 4).
    """
    robot = env.scene[asset_cfg]
    return robot.data.body_quat_w[:, -1, :]


def ee_velocity(env: ManagerBasedRLEnv, asset_cfg: str = "robot") -> torch.Tensor:
    """
    End-effector linear velocity in world frame.
    
    Args:
        env: The RL environment.
        asset_cfg: Name of the robot asset in the scene.
        
    Returns:
        End-effector velocity tensor of shape (num_envs, 3).
    """
    robot = env.scene[asset_cfg]
    return robot.data.body_lin_vel_w[:, -1, :]


def joint_positions_normalized(env: ManagerBasedRLEnv, asset_cfg: str = "robot") -> torch.Tensor:
    """
    Normalized joint positions.
    
    Args:
        env: The RL environment.
        asset_cfg: Name of the robot asset in the scene.
        
    Returns:
        Normalized joint positions tensor of shape (num_envs, num_joints).
    """
    robot = env.scene[asset_cfg]
    joint_pos = robot.data.joint_pos
    # Normalize to [-1, 1] based on joint limits
    joint_limits = robot.data.soft_joint_pos_limits
    normalized = 2.0 * (joint_pos - joint_limits[:, :, 0]) / (joint_limits[:, :, 1] - joint_limits[:, :, 0]) - 1.0
    return torch.clamp(normalized, -1.0, 1.0)


def target_position_relative(
    env: ManagerBasedRLEnv,
    target_name: str = "target",
    asset_cfg: str = "robot"
) -> torch.Tensor:
    """
    Target position relative to end-effector.
    
    Args:
        env: The RL environment.
        target_name: Name of the target object in the scene.
        asset_cfg: Name of the robot asset in the scene.
        
    Returns:
        Relative target position tensor of shape (num_envs, 3).
    """
    robot = env.scene[asset_cfg]
    ee_pos = robot.data.body_pos_w[:, -1, :]
    
    # Get target position (assuming it's stored in the environment)
    if hasattr(env, target_name):
        target = getattr(env, target_name)
        target_pos = target.data.root_pos_w
        return target_pos - ee_pos
    else:
        # Return zeros if target doesn't exist
        return torch.zeros_like(ee_pos)
