## Credit to https://github.com/JoAnnHang

import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink  #  https://github.com/kevinzakka/mink

model_path = "scene2.xml"


def get_cube_position(model, data):
    """Get the current position of the target cube"""
    try:
        cube_body_id = model.body("target_cube").id
        print(f"Debug: cube_body_id = {cube_body_id}")
        cube_pos = data.xpos[cube_body_id].copy()
        print(f"Debug: raw cube position = {cube_pos}")
        return cube_pos
    except Exception as e:
        print(f"Error getting cube position: {e}")
        # Fallback to XML position
        return np.array([0.49, 0.05, 0.23], dtype=np.float64)


def run_pick_and_place():
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # =================== #
    # Setup IK
    # =================== #
    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",  # EE site in your URDF/XML
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1e-6,
    )
    posture_task = mink.PostureTask(model, cost=1e-3)

    tasks = [end_effector_task, posture_task]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
    ]

    # Add joint velocity limits
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    limits.append(mink.VelocityLimit(model, max_velocities))

    # IK solver settings
    solver = "daqp"
    pos_threshold = 1e-3
    ori_threshold = 1e-3
    max_iters = 5

    # =================== #
    # Waypoints - will be defined dynamically based on cube position
    # =================== #
    quat = np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float64)  # 90° rotation around Y-axis to point down

    # =================== #
    # Run viewer loop
    # =================== #
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        # Print initial end effector position
        site_id = model.site("attachment_site").id
        initial_ee_pos = data.site_xpos[site_id].copy()
        print(f"Initial end effector position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")

        # Get cube position and create waypoints based on it
        cube_pos = get_cube_position(model, data)
        print(f"Initial cube position (before physics): [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        
        # Run a few simulation steps to let physics settle
        for _ in range(100):
            mujoco.mj_step(model, data)
        
        cube_pos = get_cube_position(model, data)
        print(f"Cube position (after physics settle): [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        
        # Define waypoints based on cube position
        waypoints = [
            np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15], dtype=np.float64),  # approach above cube
            np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.02], dtype=np.float64),  # move down to pick
            np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15], dtype=np.float64),  # lift cube
            np.array([0.3, -0.2, cube_pos[2] + 0.15], dtype=np.float64),                # move to place location
            np.array([0.3, -0.2, cube_pos[2] + 0.05], dtype=np.float64),                # lower to place
        ]

        posture_task.set_target_from_configuration(configuration)

        rate = RateLimiter(frequency=200.0, warn=False)

        for i, pos in enumerate(waypoints):
            print(f"Moving to waypoint {i+1}: {pos}")
            
            # Set the target marker position for visualization  
            data.mocap_pos[0] = pos  # Set mocap position to waypoint
            data.mocap_quat[0, :] = quat  # Set mocap orientation
            
            # Create SE3 transform for the target using target_marker
            T_target = mink.SE3.from_mocap_name(model, data, "target_marker")
            end_effector_task.set_target(T_target)

            # Iteratively solve IK until converged
            reached = False
            iteration_count = 0
            max_outer_iters = 20  # Prevent infinite loop
            
            while not reached and viewer.is_running() and iteration_count < max_outer_iters:
                for _ in range(max_iters):
                    vel = mink.solve_ik(
                        configuration, tasks, rate.dt, solver, limits=limits
                    )
                    configuration.integrate_inplace(vel, rate.dt)
                    err = end_effector_task.compute_error(configuration)
                    #pos_ok = np.linalg.norm(err[:3]) <= pos_threshold
                    #ori_ok = np.linalg.norm(err[3:]) <= ori_threshold
                    
                    # Debug info every 100 iterations
                    # if iteration_count % 100 == 0:
                    #     print(f"  Iteration {iteration_count}: pos_error={np.linalg.norm(err[:3]):.4f}, ori_error={np.linalg.norm(err[3:]):.4f}")
                    
                    # if pos_ok and ori_ok:
                    #     reached = True
                    #     print(f"  ✅ Reached target in {iteration_count} iterations")
                    #     break

                # Update simulation with new joint positions
                data.qpos[:6] = configuration.q[:6]  # Update joint positions
                data.ctrl[:6] = configuration.q[:6]  # Update control
                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()
                time.sleep(0.05)
                
                iteration_count += 1
                
            # if not reached:
            #     print(f"  ⚠️ Could not reach target after {iteration_count} iterations")
            
            # Print current end effector position after waypoint
            site_id = model.site("attachment_site").id
            current_ee_pos = data.site_xpos[site_id].copy()
            print(f"  End effector position: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
            print(f"  Target position:       [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            distance_error = np.linalg.norm(current_ee_pos - pos)
            print(f"  Distance error: {distance_error:.4f}m")
            print()

            #Simple open/close gripper events
            if i == 1:  # at pick (waypoint 2)
                print("🔽 Closing gripper to pick up cube...")
                print(model.actuator("fingers_actuator").id)
                data.ctrl[6] = 255.0  # Close gripper
                # Give time for gripper to close
                for _ in range(300):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    rate.sleep()
                    #time.sleep(0.02)
                for _ in range(300):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    rate.sleep()
                print("✅ Cube picked up!")
            # if i == 1:  # at pick (waypoint 2)
            #     print("🔽 Gradually closing gripper to pick up cube...")
                
            #     # Gradual gripper closing
            #     for close_step in range(10):
            #         grip_force = (close_step + 1) * 25.5  # Gradually increase from 25.5 to 255
            #         data.ctrl[6] = grip_force
            #         print(f"  Gripper force: {grip_force:.1f}")
                    
            #         # Give time for each step
            #         for _ in range(30):
            #             mujoco.mj_step(model, data)
            #             viewer.sync()
            #             rate.sleep()
            #             time.sleep(0.01)
                
            #     # Hold the grip
            #     print("  Maintaining grip...")
            #     for _ in range(100):
            #         mujoco.mj_step(model, data)
            #         viewer.sync()
            #         rate.sleep()
            #     print("✅ Cube gripped!")
            elif i == 4:  # at place (waypoint 5)
                print("🔼 Opening gripper to release cube...")
                data.ctrl[6] = 0.0  # Open gripper
                # Give time for gripper to open
                for _ in range(300):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    rate.sleep()
                print("✅ Cube released!")

        print("✅ Pick and place operation completed!")
        print("The cube has been moved from its original position.")
        time.sleep(3)


if __name__ == "__main__":
    run_pick_and_place()