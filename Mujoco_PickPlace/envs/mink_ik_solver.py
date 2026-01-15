"""
FR3 Mink Solver Module

Inverse kinematics and dynamics solver for FR3 robot using mink.
Supports task hierarchies, joint limits, and velocity constraints.

Reference: https://github.com/kevinzakka/mink
"""

import numpy as np
import mujoco as mj
import mink
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class MinkConfig:
    """Configuration for mink solver"""
    solver: str = "qp"  # "qp", "daqp"
    dt: float = 0.002  # Time step for integration
    max_iters: int = 100
    pos_threshold: float = 1e-3  # Position error threshold
    ori_threshold: float = 1e-3  # Orientation error threshold
    lm_damping: float = 1e-6  # Levenberg-Marquardt damping
    pos_cost: float = 1.0
    ori_cost: float = 1.0
    posture_cost: float = 1e-3
    frame_name: str = "end_effector"
    frame_type: str = "site"  # "site" or "body"
    verbose: bool = False


class FR3MinkSolver:
    """
    FR3 Inverse Kinematics solver using mink.
    Supports multiple control modes and task hierarchies.
    """
    
    def __init__(self, model: mj.MjModel, data: mj.MjData, config: Optional[MinkConfig] = None):
        """
        Initialize the FR3 mink solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: MinkConfig object with solver parameters
        """
        self.model = model
        self.data = data
        self.config = config or MinkConfig()
        self.n_dof = 7  # FR3 has 7 joints
        self.n_total_dof = model.nq  # Total DOF including gripper and objects
        
        # Home configuration for posture task (all DOF)
        # First 7 are FR3, next 2 are gripper, rest are object positions
        home_q = np.zeros(self.n_total_dof)
        home_q[:7] = np.array([0, -np.pi/2, 0, -3*np.pi/2, 0, np.pi/2, np.pi/2])
        # Gripper open (indices 7-8)
        home_q[7:9] = 0.04
        self.home_config = home_q
    
    def _create_configuration(self) -> mink.Configuration:
        """Create a mink configuration from current robot state"""
        config = mink.Configuration(self.model)
        # Get current state and clamp to joint limits before updating
        q = self.data.qpos.copy()
        # Only clip the joints that have ranges defined (first 12 elements)
        # The remaining DOF are free joints for objects which don't have limits
        if self.model.jnt_range.shape[0] > 0:
            n_limited = min(self.model.jnt_range.shape[0], len(q))
            q[:n_limited] = np.clip(q[:n_limited], self.model.jnt_range[:n_limited, 0], 
                                     self.model.jnt_range[:n_limited, 1])
        config.update(q)  # Use all DOF, not just first 7
        return config
    
    def _get_ee_transform(self, configuration: mink.Configuration) -> mink.SE3:
        """Get current end-effector transform from configuration"""
        if self.config.frame_type == "site":
            site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.config.frame_name)
            # Compute kinematics for this configuration
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:] = configuration.q  # Use all DOF
            mj.mj_forward(self.model, data_temp)
            
            pos = data_temp.site_xpos[site_id].copy()
            mat = data_temp.site_xmat[site_id].copy()
        else:  # body
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:] = configuration.q  # Use all DOF
            mj.mj_forward(self.model, data_temp)
            
            pos = data_temp.body(self.config.frame_name).xpos.copy()
            mat = data_temp.body(self.config.frame_name).xmat.copy()
        
        # Convert rotation matrix to SO3
        rotation = mink.SO3.from_matrix(mat.reshape(3, 3))
        return mink.SE3.from_rotation_and_translation(rotation, pos)
    
    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot_mat: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
        home_pos: Optional[np.ndarray] = None,
        apply_limits: bool = True,
        verbose: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Args:
            target_pos: Target position [x, y, z]
            target_rot_mat: Target rotation matrix (3x3), or None for orientation-free IK
            target_quat: Target quaternion [w, x, y, z], or None
            home_pos: Home position for posture task, or None to use default (all DOF)
            apply_limits: Whether to apply joint limits and velocity constraints
            verbose: Print convergence info
        Returns:
            Tuple of (joint_angles, info_dict)
        """
        info = {
            'converged': False,
            'iterations': 0,
            'final_pos_error': None,
            'final_ori_error': None,
            'solution': None
        }
        
        configuration = self._create_configuration()
        
        if home_pos is not None:
            if len(home_pos) == self.n_dof:
                # Expand to full DOF
                full_home = self.home_config.copy()
                full_home[:self.n_dof] = home_pos
                home_q = full_home
            else:
                home_q = home_pos
        else:
            home_q = self.home_config
        
        ee_task = mink.FrameTask(
            frame_name=self.config.frame_name,
            frame_type=self.config.frame_type,
            position_cost=self.config.pos_cost,
            orientation_cost=self.config.ori_cost if target_rot_mat is not None else 0.0,
            lm_damping=self.config.lm_damping,
        )
        
        # Create posture task
        posture_task = mink.PostureTask(self.model, cost=self.config.posture_cost)
        posture_task.set_target(home_q)
        
        tasks = [ee_task, posture_task]
        
        # Set target pose
        if target_rot_mat is not None:
            rotation = mink.SO3.from_matrix(target_rot_mat)
        elif target_quat is not None:
            rotation = mink.SO3(wxyz=target_quat)
        else:
            # No orientation constraint - extract rotation from current EE
            current_ee = self._get_ee_transform(configuration)
            rotation = current_ee.rotation()  # Call the method
        
        T_target = mink.SE3.from_rotation_and_translation(rotation, target_pos)
        ee_task.set_target(T_target)
        
        constraints = []
        if apply_limits:
            constraints.append(mink.ConfigurationLimit(self.model))
            constraints.append(mink.VelocityLimit(self.model))
        
        # Iterative IK solver
        for iteration in range(self.config.max_iters):
            try:
                # Solve for joint velocities
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    self.config.dt,
                    self.config.solver,
                    constraints=constraints if apply_limits else None
                )
                
                # Integrate velocities
                configuration.integrate_inplace(vel, self.config.dt)
                
                # Get current EE pose
                current_ee = self._get_ee_transform(configuration)
                ee_pos = current_ee.translation
                
                # Compute errors
                pos_error = np.linalg.norm(target_pos - ee_pos)
                
                if target_rot_mat is not None or target_quat is not None:
                    T_error = T_target.inverse() * current_ee
                    ori_error = np.linalg.norm(T_error.rotation.to_axis_angle())
                else:
                    ori_error = 0.0
                
                info['iterations'] = iteration + 1
                info['final_pos_error'] = pos_error
                info['final_ori_error'] = ori_error
                
                if verbose:
                    print(f"Iteration {iteration + 1}: pos_err={pos_error:.6f}, ori_err={ori_error:.6f}")
                
                # Check convergence
                if (pos_error <= self.config.pos_threshold and 
                    ori_error <= self.config.ori_threshold):
                    info['converged'] = True
                    # Return only FR3 joints, clamped to limits
                    q_solution = configuration.q[:self.n_dof].copy()
                    q_solution = np.clip(q_solution, self.model.jnt_range[:self.n_dof, 0], 
                                        self.model.jnt_range[:self.n_dof, 1])
                    info['solution'] = q_solution
                    if verbose:
                        print(f"✓ Converged in {iteration + 1} iterations")
                    return q_solution, info
                
            except Exception as e:
                if verbose:
                    print(f"Error at iteration {iteration}: {e}")
                continue
        
        # Return best solution found (only FR3 joints)
        if verbose:
            print(f"✗ Did not converge after {self.config.max_iters} iterations")
        q_solution = configuration.q[:self.n_dof].copy()
        q_solution = np.clip(q_solution, self.model.jnt_range[:self.n_dof, 0], 
                            self.model.jnt_range[:self.n_dof, 1])
        info['solution'] = q_solution
        return q_solution, info
    
    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for joint configuration.
        
        Args:
            q: Joint angles (7-element array)
        Returns:
            Tuple of (position, rotation_matrix)
        """
        data_temp = mj.MjData(self.model)
        data_temp.qpos[:self.n_dof] = q
        mj.mj_forward(self.model, data_temp)
        
        if self.config.frame_type == "site":
            site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.config.frame_name)
            pos = data_temp.site_xpos[site_id].copy()
            mat = data_temp.site_xmat[site_id].copy()
        else:
            pos = data_temp.body(self.config.frame_name).xpos.copy()
            mat = data_temp.body(self.config.frame_name).xmat.copy()
        
        return pos, mat.reshape(3, 3)
    
    def compute_jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            q: Joint angles, or None to use current data state
        Returns:
            6xN Jacobian matrix
        """
        if q is not None:
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:self.n_dof] = q
            mj.mj_forward(self.model, data_temp)
        else:
            data_temp = self.data
        
        if self.config.frame_type == "site":
            site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.config.frame_name)
            J_pos = np.zeros((3, self.model.nv))
            J_rot = np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, data_temp, J_pos, J_rot, site_id)
        else:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.config.frame_name)
            J_pos = np.zeros((3, self.model.nv))
            J_rot = np.zeros((3, self.model.nv))
            mj.mj_jacBody(self.model, data_temp, J_pos, J_rot, body_id)
        
        # Extract only the FR3 joints (first 7 columns)
        J_full = np.vstack([J_pos[:, :self.n_dof], J_rot[:, :self.n_dof]])
        return J_full
    
    def check_joint_limits(self, q: np.ndarray) -> Tuple[bool, List[str]]:
        """Check if joint configuration violates limits."""
        violations = []
        within_limits = True
        
        for i in range(self.n_dof):
            qmin = self.model.jnt_range[i, 0]
            qmax = self.model.jnt_range[i, 1]
            
            if q[i] < qmin or q[i] > qmax:
                within_limits = False
                violations.append(f"Joint {i}: {q[i]:.4f} outside [{qmin:.4f}, {qmax:.4f}]")
        
        return within_limits, violations
    
    def set_home_config(self, q: np.ndarray):
        """Set home configuration for posture task"""
        assert len(q) == self.n_dof
        self.home_config = q.copy()
