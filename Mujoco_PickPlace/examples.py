import numpy as np
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs.pick_place_env import PickPlaceEnv


def example_1_simple_grasp(render=False, fps=60):
    """Example: Simple grasp and release."""
    env = PickPlaceEnv(render=render, fps=fps)
    
    try:
        print("=== Example 1: Simple Grasp ===\n")
        
        home_config = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        env.controller.reset_to_config(home_config)
        env.controller.open_gripper()
        
        for _ in range(100):
            env.step(0.001)
        
        print("Approaching object obj1...")
        env.grasp_object("obj1", approach_height=0.08)
        
        print("Opening gripper and moving away...")
        env.controller.open_gripper()
        for _ in range(100):
            env.step(0.001)
        
        ee_pos, _ = env.get_ee_pose()
        print(f"Final EE position: {ee_pos}")
        print("✓ Example completed\n")
        
    finally:
        env.close()


def example_2_sequential_pick_place(render=False, fps=60):
    """Example: Pick and place multiple objects sequentially."""
    env = PickPlaceEnv(render=render, fps=fps)
    
    try:
        print("=== Example 2: Sequential Pick and Place ===\n")
        
        home_config = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        env.controller.reset_to_config(home_config)
        env.controller.open_gripper()
        
        for _ in range(100):
            env.step(0.001)
        
        objects_to_move = [
            ("obj1", np.array([-0.35, -0.3, 0.45])),
            ("obj2", np.array([-0.15, -0.3, 0.45])),
            ("obj3", np.array([0.05, -0.3, 0.45]))
        ]
        
        for obj_name, target_pos in objects_to_move:
            print(f"Moving {obj_name}...")
            env.grasp_object(obj_name, approach_height=0.08)
            env.place_object(target_pos)
            
            for _ in range(50):
                env.step(0.001)
        
        print("Returning to home...")
        env.return_home(duration=2.0)
        
        print("✓ Example completed\n")
        
    finally:
        env.close()


def example_3_state_inspection(render=False, fps=60):
    """Example: Inspect environment state."""
    env = PickPlaceEnv(render=render, fps=fps)
    
    try:
        print("=== Example 3: State Inspection ===\n")
        
        home_config = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        env.controller.reset_to_config(home_config)
        
        for _ in range(100):
            env.step(0.001)
        
        state = env.get_state()
        
        print(f"Simulation time: {state['time']:.3f}s")
        print(f"EE position: {state['ee_pos']}")
        print(f"Joint angles: {state['q']}")
        print(f"Objects:")
        for obj_name, obj_data in state["objects"].items():
            print(f"  {obj_name}: {obj_data['pos']}")
        
        objects_in_zone = env.get_object_in_zone(env.target_zone_pos)
        print(f"Objects in target zone: {objects_in_zone}")
        print("✓ Example completed\n")
        
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick and place examples")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument("--example", type=int, default=0, help="Example to run (0=all, 1-3=specific)")
    parser.add_argument("--fps", type=int, default=60, help="Visualization FPS (default: 60)")
    args = parser.parse_args()
    
    if args.example in (0, 1):
        example_1_simple_grasp(render=args.render, fps=args.fps)
    if args.example in (0, 2):
        example_2_sequential_pick_place(render=args.render, fps=args.fps)
    if args.example in (0, 3):
        example_3_state_inspection(render=args.render, fps=args.fps)
    
    print("=== All examples completed ===")
