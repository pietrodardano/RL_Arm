"""
Franka Research3 robot arm environment implementation.

This module implements the RL environment for the Franka Research3 manipulator.
"""

from __future__ import annotations

import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv

from .franka_research3_env_cfg import FrankaResearch3EnvCfg


class FrankaResearch3Env(ManagerBasedRLEnv):
    """
    Environment for Franka Research3 robot arm manipulation tasks.
    
    This environment provides a base implementation for various manipulation
    tasks using the Franka Research3 robot arm. Tasks can be customized by
    modifying the configuration and reward functions.
    """

    cfg: FrankaResearch3EnvCfg

    def __init__(self, cfg: FrankaResearch3EnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the Franka Research3 environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: The render mode for the environment.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup the scene for the Franka Research3 environment."""
        # Setup the robot in the scene
        self.scene.robot.initialize(self.scene.env_prim_paths[0] + "/Robot")
        
        # Clone the robot to all environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions within each environment
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Pre-process actions before stepping through the physics.

        Args:
            actions: The actions to apply to the robot.
        """
        # Apply actions to the robot
        self.scene.robot.set_joint_position_target(actions)

    def _get_observations(self) -> dict:
        """
        Compute and return observations.

        Returns:
            Dictionary containing observation data.
        """
        # Observations are handled by the observation manager
        obs = super()._get_observations()
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute and return rewards.

        Returns:
            Reward values for each environment.
        """
        # Rewards are handled by the reward manager
        return super()._get_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute and return termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        # Terminations are handled by the termination manager
        return super()._get_dones()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        Reset environments at the given indices.

        Args:
            env_ids: Indices of environments to reset.
        """
        super()._reset_idx(env_ids)
