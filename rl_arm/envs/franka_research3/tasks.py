"""
Example task configurations for Franka Research3 robot.

This module demonstrates how to create task-specific configurations
by extending the base FrankaResearch3EnvCfg.
"""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup

from ..franka_research3_env_cfg import FrankaResearch3EnvCfg
from rl_arm.utils import rewards, observations


@configclass
class FrankaResearch3ReachEnvCfg(FrankaResearch3EnvCfg):
    """
    Configuration for Franka Research3 reaching task.
    
    The robot must move its end-effector to reach a target position.
    """
    
    @configclass
    class ReachObservationsCfg:
        """Observations for reaching task."""
        
        @configclass
        class PolicyCfg(ObsGroup):
            """Policy observations."""
            
            # Robot state
            joint_pos = ObsTerm(func=lambda env: env.scene.robot.data.joint_pos)
            joint_vel = ObsTerm(func=lambda env: env.scene.robot.data.joint_vel)
            
            # End-effector state
            ee_pos = ObsTerm(func=observations.ee_position)
            ee_vel = ObsTerm(func=observations.ee_velocity)
            
            # Gripper state
            gripper_pos = ObsTerm(func=lambda env: env.scene.robot.data.joint_pos[:, -2:])
            
            # Target information
            target_rel_pos = ObsTerm(func=observations.target_position_relative)
            
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        
        policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class ReachRewardsCfg:
        """Rewards for reaching task."""
        
        # Main reaching reward
        reaching = RewTerm(
            func=rewards.reaching_reward,
            weight=10.0,
            params={"target_name": "target", "std": 0.1}
        )
        
        # Penalize large velocities
        joint_vel = RewTerm(
            func=rewards.joint_velocity_penalty,
            weight=0.01
        )
        
        # Penalize action rate changes
        action_rate = RewTerm(
            func=rewards.action_rate_penalty,
            weight=0.005
        )
    
    # Override base configurations
    observations: ReachObservationsCfg = ReachObservationsCfg()
    rewards: ReachRewardsCfg = ReachRewardsCfg()


@configclass
class FrankaResearch3PickAndPlaceEnvCfg(FrankaResearch3EnvCfg):
    """
    Configuration for Franka Research3 pick and place task.
    
    The robot must pick up an object and place it at a target location.
    """
    
    @configclass
    class PickAndPlaceRewardsCfg:
        """Rewards for pick and place task."""
        
        # Reward for grasping object
        grasp_success = RewTerm(
            func=rewards.reaching_reward,
            weight=8.0,
            params={"target_name": "object", "std": 0.05}
        )
        
        # Reward for lifting object
        object_height = RewTerm(
            func=lambda env: env.scene.object.data.root_pos_w[:, 2] - 0.8,
            weight=5.0
        )
        
        # Reward for placing object at target
        object_to_target = RewTerm(
            func=rewards.reaching_reward,
            weight=15.0,
            params={"target_name": "target_pos", "std": 0.1}
        )
        
        # Penalize joint velocities
        joint_vel = RewTerm(
            func=rewards.joint_velocity_penalty,
            weight=0.01
        )
        
        # Penalize being close to joint limits
        joint_limits = RewTerm(
            func=rewards.joint_limits_penalty,
            weight=0.5,
            params={"threshold": 0.95}
        )
    
    rewards: PickAndPlaceRewardsCfg = PickAndPlaceRewardsCfg()
