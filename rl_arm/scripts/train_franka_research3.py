"""
Example training script for Franka Research3 environment.

This script demonstrates how to train an RL agent on the Franka Research3 manipulation task.
"""

import argparse
import os
import torch

# IsaacLab imports (these would be actual imports in a real setup)
# from omni.isaac.lab.app import AppLauncher
# from omni.isaac.lab_tasks.utils import parse_env_cfg

# Local imports
from rl_arm.envs.franka_research3 import FrankaResearch3Env, FrankaResearch3EnvCfg
from rl_arm.configs.train.train_cfg import PPOTrainCfg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent on Franka Research3 environment")
    
    # Environment arguments
    parser.add_argument("--task", type=str, default="FrankaResearch3-Reach", help="Name of the task")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--env_spacing", type=float, default=2.5, help="Environment spacing")
    
    # Training arguments
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"], help="RL algorithm")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--experiment_name", type=str, default="franka_research3_reach", help="Experiment name")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 80)
    print(f"Training {args.algo.upper()} on {args.task}")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create environment configuration
    env_cfg = FrankaResearch3EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = args.env_spacing
    
    print(f"\nEnvironment Configuration:")
    print(f"  - Number of environments: {env_cfg.scene.num_envs}")
    print(f"  - Environment spacing: {env_cfg.scene.env_spacing}")
    print(f"  - Episode length: {env_cfg.episode_length_s}s")
    
    # Create training configuration
    train_cfg = PPOTrainCfg()
    train_cfg.num_iterations = args.num_iterations
    train_cfg.learning_rate = args.learning_rate
    train_cfg.seed = args.seed
    train_cfg.device = args.device
    
    print(f"\nTraining Configuration:")
    print(f"  - Algorithm: {args.algo.upper()}")
    print(f"  - Number of iterations: {train_cfg.num_iterations}")
    print(f"  - Learning rate: {train_cfg.learning_rate}")
    print(f"  - Device: {train_cfg.device}")
    
    # Create log directory
    log_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\nLogs will be saved to: {log_dir}")
    
    # TODO: Create environment
    # env = FrankaResearch3Env(cfg=env_cfg)
    
    # TODO: Create RL agent based on algorithm choice
    # if args.algo == "ppo":
    #     agent = PPOAgent(env, train_cfg)
    # elif args.algo == "sac":
    #     agent = SACAgent(env, train_cfg)
    # elif args.algo == "td3":
    #     agent = TD3Agent(env, train_cfg)
    
    # TODO: Training loop
    # for iteration in range(train_cfg.num_iterations):
    #     agent.train()
    #     if iteration % train_cfg.log_interval == 0:
    #         agent.log()
    #     if iteration % train_cfg.save_interval == 0:
    #         agent.save(log_dir)
    
    print("\n" + "=" * 80)
    print("Training script template created successfully!")
    print("Note: This is a template. Full implementation requires IsaacLab installation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
