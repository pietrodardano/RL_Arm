"""
Training configuration for RL agents.
"""

from dataclasses import dataclass


@dataclass
class PPOTrainCfg:
    """Configuration for PPO training."""
    
    # Training
    num_iterations: int = 1000
    num_steps_per_env: int = 24
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Policy network
    num_layers: int = 3
    hidden_dim: int = 256
    activation: str = "elu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Device
    device: str = "cuda:0"
    
    # Seed
    seed: int = 42


@dataclass
class SACTrainCfg:
    """Configuration for SAC training."""
    
    # Training
    num_iterations: int = 1000
    num_steps_per_env: int = 1
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    
    # Replay buffer
    buffer_size: int = 1000000
    
    # Policy network
    num_layers: int = 3
    hidden_dim: int = 256
    activation: str = "relu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Device
    device: str = "cuda:0"
    
    # Seed
    seed: int = 42


@dataclass
class TD3TrainCfg:
    """Configuration for TD3 training."""
    
    # Training
    num_iterations: int = 1000
    num_steps_per_env: int = 1
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    
    # Replay buffer
    buffer_size: int = 1000000
    
    # Exploration noise
    exploration_noise: float = 0.1
    
    # Policy network
    num_layers: int = 3
    hidden_dim: int = 256
    activation: str = "relu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Device
    device: str = "cuda:0"
    
    # Seed
    seed: int = 42
