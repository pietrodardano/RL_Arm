#!/usr/bin/env python3
"""Visualize the FR3 robot in the scene."""

import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
model = mujoco.MjModel.from_xml_path("Mujoco_PickPlace/assets/scene.xml")
data = mujoco.MjData(model)

# Set home configuration
home_config = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
data.qpos[:7] = home_config

# Forward pass to compute body positions
mujoco.mj_forward(model, data)

print("=" * 50)
print("FR3 Robot Kinematic Chain Visualization")
print("=" * 50)
print("\nRobot Configuration (Home):")
print(f"Joint angles: {home_config}")
print("\nBody Positions (in world frame):")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ("fr3_link" in name or "hand" in name):
        xpos = data.xpos[i]
        print(f"  {name:20s}: pos={xpos}")

print("\nOpening viewer... (Press 'q' to quit)")
print("=" * 50)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Run for visualization
    while viewer.is_running():
        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
