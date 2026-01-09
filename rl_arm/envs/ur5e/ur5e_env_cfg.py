"""
Configuration for UR5e robot arm environment.

This module defines the environment configuration for the Universal Robots UR5e
manipulator in IsaacLab.
"""

from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class UR5eSceneCfg(InteractiveSceneCfg):
    """Configuration for the UR5e scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # UR5e robot
    robot: ArticulationCfg = MISSING

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the UR5e environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Joint positions and velocities
        joint_pos = ObsTerm(func=lambda env: env.scene.robot.data.joint_pos)
        joint_vel = ObsTerm(func=lambda env: env.scene.robot.data.joint_vel)
        
        # End-effector pose
        ee_pos = ObsTerm(func=lambda env: env.scene.robot.data.body_pos_w[:, -1, :])
        ee_quat = ObsTerm(func=lambda env: env.scene.robot.data.body_quat_w[:, -1, :])

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Define observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the UR5e environment."""

    # Placeholder reward terms - to be customized based on task
    joint_vel = RewTerm(func=lambda env: -0.01 * env.scene.robot.data.joint_vel.norm(dim=-1))


@configclass
class TerminationsCfg:
    """Termination terms for the UR5e environment."""

    # Episode length termination
    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length)


@configclass
class EventCfg:
    """Configuration for events."""

    pass


@configclass
class UR5eEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UR5e manipulation environment."""

    # Scene settings
    scene: UR5eSceneCfg = UR5eSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 10.0
        
        # Simulation settings
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation
        
        # Viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
