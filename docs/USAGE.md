# Usage Guide

This guide provides detailed usage instructions and examples for the RL_Arm project.

## Basic Usage

### Training a Robot Arm

#### UR5e Robot

Train on a reaching task:
```bash
python rl_arm/scripts/train_ur5e.py \
    --task UR5e-Reach \
    --num_envs 4096 \
    --algo ppo \
    --num_iterations 1000 \
    --learning_rate 3e-4 \
    --seed 42
```

Train with custom settings:
```bash
python rl_arm/scripts/train_ur5e.py \
    --task UR5e-Reach \
    --num_envs 2048 \
    --env_spacing 3.0 \
    --algo sac \
    --num_iterations 2000 \
    --learning_rate 1e-4 \
    --headless \
    --experiment_name ur5e_custom
```

#### Franka Research3 Robot

Train on a manipulation task:
```bash
python rl_arm/scripts/train_franka_research3.py \
    --task FrankaResearch3-Reach \
    --num_envs 4096 \
    --algo ppo \
    --num_iterations 1000
```

### Evaluating Trained Policies

Evaluate a trained policy:
```bash
python rl_arm/scripts/evaluate.py \
    --task UR5e-Reach \
    --checkpoint logs/ur5e_reach/checkpoint_1000.pt \
    --num_episodes 100
```

Evaluate with video recording:
```bash
python rl_arm/scripts/evaluate.py \
    --task UR5e-Reach \
    --checkpoint logs/ur5e_reach/checkpoint_1000.pt \
    --num_episodes 50 \
    --record_video \
    --video_dir videos/ur5e_evaluation
```

## Advanced Usage

### Custom Environment Configuration

Create a custom task by extending the base environment configuration:

```python
# rl_arm/envs/ur5e/ur5e_reach_cfg.py

from rl_arm.envs.ur5e import UR5eEnvCfg
from rl_arm.utils import rewards, observations

@configclass
class UR5eReachEnvCfg(UR5eEnvCfg):
    """Configuration for UR5e reaching task."""
    
    @configclass
    class CustomRewardsCfg:
        """Custom rewards for reaching."""
        
        # Reaching reward
        reaching = RewTerm(
            func=rewards.reaching_reward,
            weight=10.0,
            params={"target_name": "target", "std": 0.1}
        )
        
        # Joint velocity penalty
        joint_vel = RewTerm(
            func=rewards.joint_velocity_penalty,
            weight=0.01
        )
        
        # Action rate penalty
        action_rate = RewTerm(
            func=rewards.action_rate_penalty,
            weight=0.005
        )
    
    rewards: CustomRewardsCfg = CustomRewardsCfg()
```

### Custom Observations

Add custom observations to your environment:

```python
from rl_arm.utils import observations

@configclass
class CustomObservationsCfg:
    """Custom observations."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""
        
        # Standard observations
        joint_pos = ObsTerm(func=lambda env: env.scene.robot.data.joint_pos)
        joint_vel = ObsTerm(func=lambda env: env.scene.robot.data.joint_vel)
        
        # Custom observations
        ee_pos = ObsTerm(func=observations.ee_position)
        ee_vel = ObsTerm(func=observations.ee_velocity)
        target_rel = ObsTerm(func=observations.target_position_relative)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
```

### Custom Reward Functions

Create domain-specific reward functions:

```python
# rl_arm/utils/custom_rewards.py

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def pick_and_place_reward(
    env: ManagerBasedRLEnv,
    object_name: str = "cube",
    target_name: str = "target",
    threshold: float = 0.05
) -> torch.Tensor:
    """Reward for pick and place task."""
    
    # Get object and target positions
    obj = env.scene[object_name]
    target = env.scene[target_name]
    
    obj_pos = obj.data.root_pos_w
    target_pos = target.data.root_pos_w
    
    # Calculate distance
    distance = torch.norm(obj_pos - target_pos, dim=-1)
    
    # Binary reward for reaching threshold
    success = (distance < threshold).float()
    
    # Dense reward component
    dense_reward = torch.exp(-distance / 0.1)
    
    return success * 10.0 + dense_reward
```

### Multi-Task Training

Train on multiple tasks sequentially or with curriculum learning:

```python
# Example: Curriculum learning script
tasks = [
    ("UR5e-Reach", 500),
    ("UR5e-Push", 500),
    ("UR5e-PickAndPlace", 1000)
]

for task_name, num_iterations in tasks:
    print(f"Training on {task_name} for {num_iterations} iterations")
    
    # Load environment
    env_cfg = load_env_cfg(task_name)
    env = UR5eEnv(cfg=env_cfg)
    
    # Train
    agent.train(env, num_iterations)
    
    # Save checkpoint
    agent.save(f"checkpoints/{task_name}.pt")
```

## Command-Line Arguments

### Training Scripts

Common arguments for `train_ur5e.py` and `train_franka_research3.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | - | Task name |
| `--num_envs` | int | 4096 | Number of parallel environments |
| `--env_spacing` | float | 2.5 | Spacing between environments |
| `--algo` | str | ppo | RL algorithm (ppo/sac/td3) |
| `--num_iterations` | int | 1000 | Number of training iterations |
| `--learning_rate` | float | 3e-4 | Learning rate |
| `--seed` | int | 42 | Random seed |
| `--device` | str | cuda:0 | Device to run on |
| `--headless` | flag | False | Run without GUI |
| `--log_dir` | str | logs | Directory for logs |
| `--experiment_name` | str | - | Experiment name |

### Evaluation Script

Arguments for `evaluate.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | - | Task name |
| `--checkpoint` | str | - | Path to checkpoint |
| `--num_envs` | int | 16 | Number of parallel environments |
| `--num_episodes` | int | 100 | Number of evaluation episodes |
| `--device` | str | cuda:0 | Device to run on |
| `--headless` | flag | False | Run without GUI |
| `--record_video` | flag | False | Record evaluation videos |
| `--video_dir` | str | videos | Directory for videos |

## Tips and Best Practices

### Performance Optimization

1. **GPU Utilization**: Monitor GPU usage with `nvidia-smi`. Aim for >80% utilization.
   
2. **Number of Environments**: Start with fewer environments and scale up:
   - Development: 64-256 environments
   - Training: 2048-8192 environments
   - Production: 4096-16384 environments

3. **Simulation Frequency**: Balance between accuracy and speed:
   - High accuracy: dt=0.005, decimation=4
   - Balanced: dt=0.01, decimation=2
   - Fast: dt=0.02, decimation=1

### Training Tips

1. **Hyperparameter Tuning**: Start with defaults and tune:
   - Learning rate: [1e-5, 1e-3]
   - Batch size: [256, 2048]
   - Discount factor: [0.95, 0.99]

2. **Reward Shaping**: Design rewards that are:
   - Dense (provide feedback at every step)
   - Bounded (avoid extremely large values)
   - Balanced (multiple objectives weighted appropriately)

3. **Curriculum Learning**: Start with easier tasks and progressively increase difficulty

### Debugging

1. **Check observations**: Print observation shapes and values
   ```python
   obs = env.reset()
   print(f"Observation shape: {obs['policy'].shape}")
   print(f"Observation range: [{obs['policy'].min()}, {obs['policy'].max()}]")
   ```

2. **Visualize rewards**: Log reward components separately
   
3. **Monitor training**: Use TensorBoard or Weights & Biases
   ```bash
   tensorboard --logdir logs/
   ```

## Examples

### Example 1: Reaching Task

```bash
# Train UR5e to reach a target position
python rl_arm/scripts/train_ur5e.py \
    --task UR5e-Reach \
    --num_envs 4096 \
    --algo ppo \
    --num_iterations 500 \
    --experiment_name ur5e_reaching
```

### Example 2: Multi-GPU Training

```bash
# Train on multiple GPUs (requires distributed setup)
torchrun --nproc_per_node=4 rl_arm/scripts/train_ur5e.py \
    --task UR5e-Reach \
    --num_envs 16384 \
    --algo ppo
```

### Example 3: Hyperparameter Sweep

```bash
# Sweep over learning rates
for lr in 1e-4 5e-4 1e-3; do
    python rl_arm/scripts/train_ur5e.py \
        --task UR5e-Reach \
        --learning_rate $lr \
        --experiment_name ur5e_lr_${lr}
done
```

## Next Steps

- Check [API.md](API.md) for detailed API reference
- See [SETUP.md](SETUP.md) for installation troubleshooting
- Explore the codebase for more examples
