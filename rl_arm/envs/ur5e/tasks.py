"""
Example task configurations for UR5e robot.

This module demonstrates how to create task-specific configurations
by extending the base UR5eEnvCfg.
"""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup

from ..ur5e_env_cfg import UR5eEnvCfg
from rl_arm.utils import rewards, observations


@configclass
class UR5eReachEnvCfg(UR5eEnvCfg):
    """
    Configuration for UR5e reaching task.
    
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
class UR5ePushEnvCfg(UR5eEnvCfg):
    """
    Configuration for UR5e pushing task.
    
    The robot must push an object to a target location.
    """
    
    @configclass
    class PushRewardsCfg:
        """Rewards for pushing task."""
        
        # Reward for object reaching target
        object_to_target = RewTerm(
            func=rewards.reaching_reward,
            weight=15.0,
            params={"target_name": "target_pos", "std": 0.1}
        )
        
        # Reward for end-effector approaching object
        ee_to_object = RewTerm(
            func=rewards.reaching_reward,
            weight=5.0,
            params={"target_name": "object", "std": 0.15}
        )
        
        # Penalize joint velocities
        joint_vel = RewTerm(
            func=rewards.joint_velocity_penalty,
            weight=0.01
        )
    
    rewards: PushRewardsCfg = PushRewardsCfg()
