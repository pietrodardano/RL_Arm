import numpy as np
import mujoco as mj
from typing import Optional, Tuple

# Import local mink solver
try:
    from .mink_ik_solver import FR3MinkSolver, MinkConfig
except ImportError:
    FR3MinkSolver = None
    MinkConfig = None


class RobotController:
    """Robot interface with mink IK solver integration."""
    
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        self.model = model
        self.data = data
        self.n_dof = 7
        self.n_fingers = 2
        
        # PD control gains for joint tracking
        self.kp = 100.0
        self.kd = 20.0
        self.target_q = np.zeros(self.n_dof)
        
        # Initialize mink solver if available
        if FR3MinkSolver is not None:
            mink_config = MinkConfig(
                solver="qp",
                dt=0.002,
                max_iters=100,
                pos_threshold=1e-3,
                ori_threshold=1e-3,
                frame_name="end_effector",
                frame_type="site",
                verbose=False
            )
            self.mink_solver = FR3MinkSolver(self.model, self.data, mink_config)
        else:
            self.mink_solver = None
        
    def set_joint_targets(self, q_targets: np.ndarray):
        """Set target joint angles."""
        assert len(q_targets) == self.n_dof
        self.target_q = q_targets.copy()
    
    def maintain_targets(self):
        """Apply PD control to reach targets. Called each simulation step."""
        q = self.data.qpos[:self.n_dof]
        dq = self.data.qvel[:self.n_dof]
        
        for i in range(self.n_dof):
            self.data.ctrl[i] = self.kp * (self.target_q[i] - q[i]) - self.kd * dq[i]
    
    def set_gripper(self, width: float):
        """Set gripper: 0=closed, 0.04=open.
        For slide joints, set the control value directly (not width/2)."""
        if self.model.nu >= 8:
            # Gripper uses general actuator at index 7 for left_finger
            # Equality constraint keeps right_finger synced
            self.data.ctrl[7] = width
    
    def close_gripper(self):
        self.set_gripper(0.0)
    
    def open_gripper(self):
        self.set_gripper(0.04)
    
    def get_ee_pose(self) -> tuple:
        """Get EE position and rotation matrix."""
        ee_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "end_effector")
        pos = self.data.site_xpos[ee_id].copy()
        rot = self.data.site_xmat[ee_id].reshape(3, 3).copy()
        return pos, rot
    
    def get_ee_pos(self) -> np.ndarray:
        """Get EE position."""
        return self.get_ee_pose()[0]
    
    def get_ee_quat(self) -> np.ndarray:
        """Get EE quaternion [x,y,z,w]."""
        ee_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "end_effector")
        quat = np.zeros(4)
        mat = self.data.site_xmat[ee_id]  # 9-element array
        mj.mju_mat2Quat(quat, mat)
        return quat
    
    def get_q(self) -> np.ndarray:
        """Get joint angles."""
        return self.data.qpos[:self.n_dof].copy()
    
    def get_dq(self) -> np.ndarray:
        """Get joint velocities."""
        return self.data.qvel[:self.n_dof].copy()
    
    def reset_q(self, q: np.ndarray):
        """Reset joint configuration."""
        self.data.qpos[:self.n_dof] = q
        self.target_q = q.copy()
        mj.mj_forward(self.model, self.data)
    
    # ==================== MINK IK METHODS ====================
    
    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot_mat: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
        max_iters: int = 100,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-3,
        verbose: bool = False
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Solve inverse kinematics for target position and orientation using mink.
        
        Args:
            target_pos: Target position [x, y, z]
            target_rot_mat: Target rotation matrix (3x3), or None for position-only IK
            target_quat: Target quaternion [w, x, y, z], alternative to target_rot_mat
            max_iters: Maximum number of solver iterations
            pos_threshold: Position error convergence threshold
            ori_threshold: Orientation error convergence threshold
            verbose: Print convergence information
            
        Returns:
            Tuple of (joint_angles, info_dict) where:
                - joint_angles: 7-element array of joint angles, or None if failed
                - info_dict: Dictionary with convergence information
        """
        if self.mink_solver is None:
            print("Warning: mink solver not available. Cannot solve IK.")
            return None, {'error': 'mink_solver_not_available'}
        
        # Update solver configuration
        self.mink_solver.config.max_iters = max_iters
        self.mink_solver.config.pos_threshold = pos_threshold
        self.mink_solver.config.ori_threshold = ori_threshold
        self.mink_solver.config.verbose = verbose
        
        # Solve IK
        q_solution, info = self.mink_solver.inverse_kinematics(
            target_pos=target_pos,
            target_rot_mat=target_rot_mat,
            target_quat=target_quat,
            apply_limits=True,
            verbose=verbose
        )
        
        return q_solution, info
    
    def inverse_kinematics_and_set_target(
        self,
        target_pos: np.ndarray,
        target_rot_mat: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
        max_iters: int = 100,
        verbose: bool = False
    ) -> bool:
        """
        Solve IK and automatically set as target for joint control.
        
        Args:
            target_pos: Target position
            target_rot_mat: Target rotation matrix
            target_quat: Target quaternion
            max_iters: Maximum solver iterations
            verbose: Print info
            
        Returns:
            True if IK converged, False otherwise
        """
        q_solution, info = self.inverse_kinematics(
            target_pos,
            target_rot_mat=target_rot_mat,
            target_quat=target_quat,
            max_iters=max_iters,
            verbose=verbose
        )
        
        if q_solution is not None and info.get('converged', False):
            self.set_joint_targets(q_solution)
            return True
        return False
    
    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics.
        
        Args:
            q: Joint angles (7-element array)
            
        Returns:
            Tuple of (position, rotation_matrix)
        """
        if self.mink_solver is None:
            print("Warning: mink solver not available.")
            return None, None
        
        return self.mink_solver.forward_kinematics(q)
    
    def compute_jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Jacobian matrix.
        
        Args:
            q: Joint angles, or None to use current state
            
        Returns:
            6x7 Jacobian matrix
        """
        if self.mink_solver is None:
            print("Warning: mink solver not available.")
            return None
        
        return self.mink_solver.compute_jacobian(q)
    
    def cartesian_velocity_to_joint_velocity(
        self,
        cart_vel: np.ndarray,
        q: Optional[np.ndarray] = None,
        damping: float = 0.01
    ) -> np.ndarray:
        """
        Convert Cartesian velocity to joint velocity using damped pseudo-inverse.
        
        Args:
            cart_vel: Cartesian velocity [vx, vy, vz, wx, wy, wz]
            q: Joint configuration, or None for current state
            damping: Damping for pseudo-inverse (regularization)
            
        Returns:
            Joint velocities (7-element array)
        """
        J = self.compute_jacobian(q)
        if J is None:
            return np.zeros(self.n_dof)
        
        # Damped pseudo-inverse
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
        q_vel = J_pinv @ cart_vel
        
        return q_vel
    
    def solve_cartesian_trajectory(
        self,
        waypoints: list,
        num_steps_between: int = 10,
        verbose: bool = False
    ) -> Tuple[Optional[np.ndarray], list]:
        """
        Solve IK for a sequence of Cartesian waypoints.
        
        Args:
            waypoints: List of (pos, rot_mat) tuples
            num_steps_between: Steps to interpolate between waypoints
            verbose: Print convergence info
            
        Returns:
            Tuple of (trajectory_array, info_list)
        """
        if self.mink_solver is None:
            print("Warning: mink solver not available.")
            return None, []
        
        return self.mink_solver.solve_cartesian_trajectory(
            waypoints,
            num_steps_between=num_steps_between
        )
    
    def check_joint_limits(self, q: np.ndarray) -> Tuple[bool, list]:
        """Check if joint configuration violates limits."""
        if self.mink_solver is None:
            return True, []
        return self.mink_solver.check_joint_limits(q)
