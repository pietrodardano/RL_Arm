import numpy as np
from dataclasses import dataclass


@dataclass
class RobotConfig:
    """FR3 robot configuration."""
    n_dof: int = 7
    n_fingers: int = 2
    home_q: np.ndarray = None
    
    def __post_init__(self):
        if self.home_q is None:
            self.home_q = np.array([0, -0.3854, 0, -2.3562, 0, 1.2708, 0.3854])  # self.home_q = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])


@dataclass
class GraspConfig:
    """Grasping parameters."""
    approach_height: float = 0.08
    grasp_duration: float = 1.5
    place_approach_height: float = 0.1
    placement_duration: float = 1.0
    gripper_close_steps: int = 50
    gripper_open_steps: int = 50


@dataclass
class SceneConfig:
    """Scene configuration."""
    object_names: list = None
    target_zone_pos: np.ndarray = None
    target_zone_size: float = 0.15
    physics_timestep: float = 0.001
    
    def __post_init__(self):
        if self.object_names is None:
            self.object_names = ["obj1", "obj2", "obj3"]
        if self.target_zone_pos is None:
            self.target_zone_pos = np.array([-0.3, -0.3, 0.5])


@dataclass
class SimulationConfig:
    """Full simulation configuration."""
    robot: RobotConfig = None
    grasp: GraspConfig = None
    scene: SceneConfig = None
    render: bool = False
    
    def __post_init__(self):
        if self.robot is None:
            self.robot = RobotConfig()
        if self.grasp is None:
            self.grasp = GraspConfig()
        if self.scene is None:
            self.scene = SceneConfig()


def get_default_config() -> SimulationConfig:
    """Get default simulation configuration."""
    return SimulationConfig()
