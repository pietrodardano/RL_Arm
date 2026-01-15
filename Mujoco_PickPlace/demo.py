import numpy as np
import sys
from pathlib import Path
import argparse
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs.pick_place_env import PickPlaceEnv


def main(render=False, fps=30):
    env = PickPlaceEnv(render=render, fps=fps)
    
    try:
        home_config = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        env.controller.reset_q(home_config)
        env.controller.open_gripper()
        
        print("Letting simulation settle...")
        for _ in range(100):
            env.step(0.001)
        
        print("\n=== Pick and Place Demonstration ===")
        print(f"Objects in scene: {env.object_names}")
        
        tasks = [
            ("obj1", np.array([-0.3, -0.3, 0.45])),
            ("obj2", np.array([-0.2, -0.3, 0.45])),
            ("obj3", np.array([-0.1, -0.3, 0.45]))
        ]
        
        for task_idx, (obj_name, target_pos) in enumerate(tasks, 1):
            print(f"\nTask {task_idx}: Pick {obj_name} and place at target zone")
            
            state_before = env.get_state()
            obj_pos_before = state_before["objects"][obj_name]["pos"]
            print(f"  Object position: {obj_pos_before}")
            
            #time.sleep(3.0)
            try:
                env.grasp_object(obj_name, approach_height=0.08)
                print(f"  ✓ Grasped {obj_name}")
            except Exception as e:
                print(f"  ✗ Failed to grasp: {e}")
                continue
            
            try:
                env.place_object(target_pos, approach_height=0.1)
                print(f"  ✓ Placed {obj_name} at target")
            except Exception as e:
                print(f"  ✗ Failed to place: {e}")
                continue
            
            # Wait for object to settle
            for _ in range(200):
                env.step(0.001)
            
            state_after = env.get_state()
            objects_in_zone = env.get_object_in_zone(env.target_zone_pos)
            print(f"  Objects in target zone: {objects_in_zone}")
        
        print("\n=== Returning to home ===")
        env.return_home(num_steps=50)
        
        for _ in range(200):
            env.step(0.001)
        
        final_state = env.get_state()
        print(f"\n=== Summary ===")
        print(f"Simulation time: {final_state['time']:.2f}s")
        print(f"Final EE position: {final_state['ee_pos']}")
        
        # Show final object positions
        print("\nFinal object positions:")
        for obj_name in env.object_names:
            obj_pos = final_state["objects"][obj_name]["pos"]
            print(f"  {obj_name}: {obj_pos}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick and place demonstration")
    parser.add_argument("--render", action="store_true", default=True, help="Enable visualization")
    parser.add_argument("--fps", type=int, default=60, help="Visualization FPS (default: 60)")
    args = parser.parse_args()
    main(render=args.render, fps=args.fps)
