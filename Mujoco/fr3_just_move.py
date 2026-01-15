import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

from robot_descriptions import fr3_mj_description
from dm_control import mjcf
import mink
import tempfile
import os
import xml.etree.ElementTree as ET

# Use the scene.xml from assets which includes FR3 with environment
scene_xml_path = '/home/user/Documents/RL_Arm/assets/fr3/_mujoco/franka_fr3/scene.xml'
MODEL = mujoco.MjModel.from_xml_path(scene_xml_path)

class FR3Controller:
    """Simple FR3 controller for point-to-point motion planning"""

    def __init__(self, xml_path="null"):
        """Initialize the FR3 robot in MuJoCo"""
        self.model = MODEL # mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Control parameters
        self.dt = 0.01  # 10ms timestep for smoother visualization
        self.kp = 1.0   # Position gain (further reduced)
        self.kd = 0.5    # Damping gain (further reduced)
        
        # Velocity limits (rad/s)
        # A1-A4: 150 °/s ≈ 2.618 rad/s, A5-A7: 301 °/s ≈ 5.253 rad/s
        self.max_vel = np.array([2.618, 2.618, 2.618, 2.618, 5.253, 5.253, 5.253])*0.2
        
        # Get indices
        self.n_joints = 7 
        
        # Home position (ready pose)
        self.home_pos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        
    def move_to_position(self, target_pos, duration=2.0, max_steps=None, viewer=None, sync_interval=10):
        """
        Move robot to target joint position with PD control and velocity limits
        
        Args:
            target_pos: Target joint positions (7 values)
            duration: Time to reach target in seconds
            max_steps: Override duration with explicit step count
            viewer: Mujoco viewer for real-time rendering
            sync_interval: How often to sync the viewer (every N steps)
        """
        if max_steps is None:
            max_steps = int(duration / self.dt)
        
        trajectory = []
        
        for step in range(max_steps):
            # Linear interpolation
            alpha = step / max_steps
            desired_pos = self.home_pos * (1 - alpha) + target_pos * alpha
            
            # PD control with velocity limits
            error = desired_pos - self.data.qpos[:self.n_joints]
            error_vel = -self.data.qvel[:self.n_joints]
            
            desired_vel = self.kp * error
            desired_vel = np.clip(desired_vel, -self.max_vel, self.max_vel)
            
            torques = desired_vel + self.kd * error_vel
            self.data.ctrl[:self.n_joints] = torques
            
            # No gripper control
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            trajectory.append(self.data.qpos[:self.n_joints].copy())
            
            # Sync viewer for real-time rendering
            if viewer and step % sync_interval == 0:
                viewer.sync()
        
        return np.array(trajectory)
    
    def grasp(self, duration=1.0):
        """Close gripper to grasp object"""
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.data.ctrl[:self.n_joints] = np.zeros(self.n_joints)  # Hold position
            # No gripper control
            mujoco.mj_step(self.model, self.data)
    
    def release(self, duration=0.5):
        """Open gripper to release object"""
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.data.ctrl[:self.n_joints] = np.zeros(self.n_joints)  # Hold position
            # No gripper control
            mujoco.mj_step(self.model, self.data)
    
    def get_ee_pose(self):
        """Get end-effector position and orientation"""
        site_id = self.model.site("attachment_site").id
        pos = self.data.site_xpos[site_id].copy()
        # For orientation, get the 3x3 rotation matrix and convert to quaternion
        xmat = self.data.site_xmat[site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())
        return pos, quat
    
    def inverse_kinematics(self, target_pos, target_quat, max_iters=100, pos_threshold=1e-3, ori_threshold=1e-3):
        """
        Compute inverse kinematics for FR3 to reach target pose using mink.
        
        Args:
            target_pos: Target position (x, y, z) as numpy array
            target_quat: Target orientation as quaternion (w, x, y, z) as numpy array
            max_iters: Maximum number of iterations for IK solver
            pos_threshold: Position error threshold for convergence
            ori_threshold: Orientation error threshold for convergence
            
        Returns:
            joint_positions: 7-element numpy array of joint angles, or None if not converged
        """
        # Create mink configuration
        configuration = mink.Configuration(self.model)
        configuration.update(self.data.qpos)
        
        # Define tasks
        end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        )
        
        posture_task = mink.PostureTask(self.model, cost=1e-3)
        posture_task.set_target_from_configuration(configuration)
        
        tasks = [end_effector_task, posture_task]
        
        # Set up limits
        limits = [
            mink.ConfigurationLimit(self.model),
        ]
        
        # Create target SE3 transform
        T_target = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=target_quat),
            translation=target_pos
        )
        end_effector_task.set_target(T_target)
        
        # Solve IK
        solver = "daqp"
        dt = 0.02  # Time step for integration
        
        for _ in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits)
            configuration.integrate_inplace(vel, dt)
            
            # Check convergence
            err = end_effector_task.compute_error(configuration)
            pos_error = np.linalg.norm(err[:3])
            ori_error = np.linalg.norm(err[3:])
            
            if pos_error <= pos_threshold and ori_error <= ori_threshold:
                return configuration.q.copy()
        
        # If not converged, return None
        return None
    
    def reset(self):
        """Reset robot to home position"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = self.home_pos


def ik_demo():
    """Demo: move robot between two points using inverse kinematics"""
    
    # Initialize
    controller = FR3Controller()
    controller.reset()
    
    # Launch viewer for visualization
    viewer = mujoco.viewer.launch_passive(controller.model, controller.data)
    
    print("FR3 Inverse Kinematics Demo")
    print("=" * 30)
    
    # Define target poses (position + quaternion)
    # Point A: forward reach
    target_pos_a = np.array([0.4, 0.0, 0.2])
    target_quat_a = np.array([0.7071, 0.0, 0.7071, 0.0])  # w,x,y,z for 90° rotation around Y-axis (pointing down)
    
    # Point B: side reach
    target_pos_b = np.array([0.4, 0.15, 0.25])
    target_quat_b = np.array([0.7071, 0.0, 0.7071, 0.0])  # Same orientation
    
    time.sleep(2)  # Wait before starting
    
    print("Initial end-effector pose:")
    pos, quat = controller.get_ee_pose()
    print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"  Orientation: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
    
    print("\n1. Computing IK for point A...")
    q_a = controller.inverse_kinematics(target_pos_a, target_quat_a)
    if q_a is None:
        print("  ❌ IK failed for point A")
        return
    print(f"  ✅ Found joint positions: {q_a}")
    
    print("\n2. Moving to point A...")
    controller.move_to_position(q_a, duration=8.0, viewer=viewer)
    
    print("3. Computing IK for point B...")
    q_b = controller.inverse_kinematics(target_pos_b, target_quat_b)
    if q_b is None:
        print("  ❌ IK failed for point B")
        return
    print(f"  ✅ Found joint positions: {q_b}")
    
    print("\n4. Moving to point B...")
    controller.move_to_position(q_b, duration=8.0, viewer=viewer)
    
    print("\n5. Moving back to point A...")
    controller.move_to_position(q_a, duration=8.0, viewer=viewer)
    
    print("\n6. Moving back to home...")
    controller.move_to_position(controller.home_pos, duration=8.0, viewer=viewer)
    
    print("\n✓ IK demo complete!")
    print("Viewer will close in 5 seconds...")
    time.sleep(5)


if __name__ == "__main__":
    ik_demo()