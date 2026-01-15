import numpy as np
import mujoco as mj
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_controller import RobotController
from mink_ik_solver import FR3MinkSolver, MinkConfig
from utils.math_utils import get_object_pose
from utils.visualizer import Visualizer


class PickPlaceEnv:
    def __init__(self, render: bool = True, physics_timestep: float = 0.001, fps: int = 60):
        script_dir = Path(__file__).parent.parent
        xml_path = script_dir / "assets" / "scene.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene file not found: {xml_path}")
        
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.controller = RobotController(self.model, self.data)
        
        self.timestep = physics_timestep
        self.render = render
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_step_time = time.time()
        self.visualizer = None
        
        if render:
            self.visualizer = Visualizer(self.model, self.data, fps=fps)
            self.visualizer.create()
        
        self.object_names = ["obj1", "obj2", "obj3"]
        self.target_zone_pos = np.array([-0.3, -0.3, 0.5])
        
        # Grasp state tracking
        self.grasped_object = None
        self.grasped_object_offset = None
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        mj.mj_resetData(self.model, self.data)
        
        home_config = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        self.controller.reset_q(home_config)
        self.controller.open_gripper()
    
    def step(self, dt: float = None):
        """Step simulation and enforce frame rate."""
        if dt is None:
            dt = self.timestep
        
        steps = int(dt / self.timestep)
        for _ in range(steps):
            self.controller.maintain_targets()
            
            # If we have a grasped object, move it with the end effector
            if self.grasped_object is not None:
                ee_pos = self.controller.get_ee_pos()
                target_obj_pos = ee_pos + self.grasped_object_offset
                qpos_start = self.grasped_object['qpos_start']
                # Only update position, keep orientation
                self.data.qpos[qpos_start:qpos_start + 3] = target_obj_pos
            
            mj.mj_step(self.model, self.data)
        
        if self.visualizer:
            self.visualizer.sync()
        else:
            current_time = time.time()
            elapsed = current_time - self.last_step_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
            self.last_step_time = time.time()
    
    def get_ee_pose(self) -> tuple:
        """Get end-effector pose."""
        return self.controller.get_ee_pose()
    
    def set_joint_targets(self, q_targets: np.ndarray):
        """Set target joint angles."""
        self.controller.set_joint_targets(q_targets)
    
    def move_to_pose(self, target_pos: np.ndarray, target_rot: np.ndarray = None,
                     num_steps: int = 20):
        """Move EE to target pose using mink IK."""
        # Create mink solver
        mink_config = MinkConfig(max_iters=100, verbose=False)
        ik_solver = FR3MinkSolver(self.model, self.data, mink_config)
        
        # Solve IK for target pose
        q_solution, info = ik_solver.inverse_kinematics(
            target_pos,
            target_rot_mat=target_rot,
            apply_limits=True,
            verbose=False
        )
        
        if q_solution is None:
            print(f"Warning: IK failed to converge for target {target_pos}")
            return
        
        # Linear interpolation in joint space from current to target
        current_q = self.controller.get_q()
        for step in range(num_steps):
            alpha = step / num_steps
            q_interp = (1 - alpha) * current_q + alpha * q_solution
            self.controller.set_joint_targets(q_interp)
            self.step()
    
    def grasp_object(self, obj_name: str, approach_height: float = 0.05):
        """Pick an object from table by creating a temporary weld constraint."""
        obj_pos, _ = get_object_pose(self.model, self.data, obj_name)
        
        approach_pos = obj_pos.copy()
        approach_pos[2] += approach_height
        
        self.controller.open_gripper()
        self.move_to_pose(approach_pos, num_steps=20)
        
        for _ in range(50):
            self.step()
        
        # Move down to object
        self.move_to_pose(obj_pos, num_steps=15)
        
        for _ in range(20):
            self.step()
        
        # Close gripper
        self.controller.close_gripper()
        
        for _ in range(100):
            self.step()
        
        # Find the qpos index for this object's free joint
        # Objects are ordered as obj1, obj2, obj3
        obj_idx = self.object_names.index(obj_name)
        # qpos structure: 0-6 (arm), 7-8 (gripper), then 7 DOF per free joint (3 pos + 4 quat)
        qpos_start = 9 + obj_idx * 7
        
        # Store grasp info for later release
        self.grasped_object = {
            'name': obj_name,
            'qpos_start': qpos_start
        }
        
        # Attach object to hand through position control
        self.grasped_object_offset = obj_pos - self.controller.get_ee_pos()
        
        # Lift object
        lift_pos = obj_pos.copy()
        lift_pos[2] += 0.15
        self.move_to_pose(lift_pos, num_steps=20)
    
    def place_object(self, place_pos: np.ndarray, approach_height: float = 0.1):
        """Place object at location."""
        if self.grasped_object is None:
            print("Warning: No grasped object to place")
            return
        
        approach_pos = place_pos.copy()
        approach_pos[2] += approach_height
        
        self.move_to_pose(approach_pos, num_steps=20)
        
        for _ in range(50):
            self.step()
        
        self.move_to_pose(place_pos, num_steps=15)
        
        for _ in range(20):
            self.step()
        
        # Release the grasped object by updating its position directly to target
        qpos_start = self.grasped_object['qpos_start']
        release_pos = place_pos.copy()
        release_pos[2] = max(release_pos[2], 0.48)  # Ensure above ground
        self.data.qpos[qpos_start:qpos_start + 3] = release_pos
        
        # Open gripper
        self.controller.open_gripper()
        
        # Clear grasp state
        self.grasped_object = None
        self.grasped_object_offset = None
        
        for _ in range(50):
            self.step()
    
    def return_home(self, num_steps: int = 50):
        """Return robot to home configuration."""
        home_config = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        self.controller.open_gripper()
        
        # Linear interpolation to home configuration
        current_q = self.controller.get_q()
        for step in range(num_steps):
            alpha = step / num_steps
            q_interp = (1 - alpha) * current_q + alpha * home_config
            self.controller.set_joint_targets(q_interp)
            self.step()
    
    def get_object_in_zone(self, zone_pos: np.ndarray, zone_size: float = 0.15) -> list:
        """Get objects within target zone."""
        in_zone = []
        for obj_name in self.object_names:
            obj_pos, _ = get_object_pose(self.model, self.data, obj_name)
            dist_xy = np.sqrt((obj_pos[0] - zone_pos[0])**2 + 
                             (obj_pos[1] - zone_pos[1])**2)
            if dist_xy < zone_size:
                in_zone.append(obj_name)
        return in_zone
    
    def get_state(self) -> dict:
        """Get environment state."""
        state = {
            "time": self.data.time,
            "ee_pos": self.controller.get_ee_pos(),
            "ee_rot": self.controller.get_ee_pose()[1],
            "q": self.controller.get_q(),
            "dq": self.controller.get_dq(),
            "objects": {}
        }
        
        for obj_name in self.object_names:
            pos, rot = get_object_pose(self.model, self.data, obj_name)
            state["objects"][obj_name] = {"pos": pos, "rot": rot}
        
        return state
    
    def toggle_visualization(self, enable: bool = None):
        """Toggle visualization on/off."""
        if enable is None:
            enable = self.visualizer is None
        
        if enable and self.visualizer is None:
            self.visualizer = Visualizer(self.model, self.data, fps=self.fps)
            self.visualizer.create()
            self.render = True
        elif not enable and self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None
            self.render = False
    
    def set_fps(self, fps: int):
        """Set visualization and simulation FPS."""
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_step_time = time.time()
        if self.visualizer:
            self.visualizer.set_fps(fps)
    
    def close(self):
        """Close environment."""
        if self.visualizer:
            self.visualizer.close()
