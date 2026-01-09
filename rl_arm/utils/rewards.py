"""
Custom reward functions for robot arm manipulation tasks.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reaching_reward(
    env: ManagerBasedRLEnv,
    target_name: str = "target",
    asset_cfg: str = "robot",
    std: float = 0.1,
) -> torch.Tensor:
    """
    Reward for reaching a target position.
    
    Uses a Gaussian kernel to provide dense rewards based on distance to target.
    
    Args:
        env: The RL environment.
        target_name: Name of the target object in the scene.
        asset_cfg: Name of the robot asset in the scene.
        std: Standard deviation for the Gaussian kernel.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot = env.scene[asset_cfg]
    ee_pos = robot.data.body_pos_w[:, -1, :]
    
    if hasattr(env, target_name):
        target = getattr(env, target_name)
        target_pos = target.data.root_pos_w
        distance = torch.norm(target_pos - ee_pos, dim=-1)
        reward = torch.exp(-(distance ** 2) / (2 * std ** 2))
        return reward
    else:
        return torch.zeros(env.num_envs, device=env.device)


def reaching_position_l2(
    env: ManagerBasedRLEnv,
    target_name: str = "target",
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """
    Negative L2 distance to target position.
    
    Args:
        env: The RL environment.
        target_name: Name of the target object in the scene.
        asset_cfg: Name of the robot asset in the scene.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot = env.scene[asset_cfg]
    ee_pos = robot.data.body_pos_w[:, -1, :]
    
    if hasattr(env, target_name):
        target = getattr(env, target_name)
        target_pos = target.data.root_pos_w
        distance = torch.norm(target_pos - ee_pos, dim=-1)
        return -distance
    else:
        return torch.zeros(env.num_envs, device=env.device)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: str = "robot",
    scale: float = 0.01,
) -> torch.Tensor:
    """
    Penalty for large joint velocities (encourages smooth motion).
    
    Args:
        env: The RL environment.
        asset_cfg: Name of the robot asset in the scene.
        scale: Scaling factor for the penalty.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot = env.scene[asset_cfg]
    joint_vel = robot.data.joint_vel
    return -scale * torch.sum(joint_vel ** 2, dim=-1)


def action_rate_penalty(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
) -> torch.Tensor:
    """
    Penalty for large action changes (encourages smooth control).
    
    Args:
        env: The RL environment.
        scale: Scaling factor for the penalty.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    if hasattr(env, "actions") and hasattr(env, "previous_actions"):
        action_diff = env.actions - env.previous_actions
        return -scale * torch.sum(action_diff ** 2, dim=-1)
    else:
        return torch.zeros(env.num_envs, device=env.device)


def joint_limits_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: str = "robot",
    threshold: float = 0.95,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Penalty for being close to joint limits.
    
    Args:
        env: The RL environment.
        asset_cfg: Name of the robot asset in the scene.
        threshold: Threshold (as fraction of range) for applying penalty.
        scale: Scaling factor for the penalty.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot = env.scene[asset_cfg]
    joint_pos = robot.data.joint_pos
    joint_limits = robot.data.soft_joint_pos_limits
    
    # Normalize joint positions to [0, 1]
    normalized = (joint_pos - joint_limits[:, :, 0]) / (joint_limits[:, :, 1] - joint_limits[:, :, 0])
    
    # Apply penalty when outside threshold
    lower_violation = torch.clamp(threshold - normalized, min=0.0)
    upper_violation = torch.clamp(normalized - (1.0 - threshold), min=0.0)
    
    penalty = torch.sum(lower_violation + upper_violation, dim=-1)
    return -scale * penalty


def orientation_alignment_reward(
    env: ManagerBasedRLEnv,
    target_name: str = "target",
    asset_cfg: str = "robot",
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Reward for aligning end-effector orientation with target orientation.
    
    Args:
        env: The RL environment.
        target_name: Name of the target object in the scene.
        asset_cfg: Name of the robot asset in the scene.
        scale: Scaling factor for the reward.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot = env.scene[asset_cfg]
    ee_quat = robot.data.body_quat_w[:, -1, :]
    
    if hasattr(env, target_name):
        target = getattr(env, target_name)
        target_quat = target.data.root_quat_w
        
        # Compute quaternion dot product (measure of alignment)
        dot_product = torch.sum(ee_quat * target_quat, dim=-1)
        alignment = 2 * dot_product ** 2 - 1  # Convert to [-1, 1]
        
        return scale * alignment
    else:
        return torch.zeros(env.num_envs, device=env.device)
