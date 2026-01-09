# API Reference

Complete API documentation for the RL_Arm package.

## Package Structure

- `rl_arm.envs` - Environment definitions
- `rl_arm.agents` - RL agent implementations
- `rl_arm.configs` - Configuration modules
- `rl_arm.scripts` - Training and evaluation scripts
- `rl_arm.utils` - Utility functions

## Environments

### UR5e Environment

#### `rl_arm.envs.ur5e.UR5eEnv`

Environment for UR5e robot arm manipulation tasks.

**Constructor:**
```python
UR5eEnv(cfg: UR5eEnvCfg, render_mode: str | None = None, **kwargs)
```

**Parameters:**
- `cfg` (UR5eEnvCfg): Environment configuration
- `render_mode` (str, optional): Render mode ("human", "rgb_array", None)
- `**kwargs`: Additional arguments passed to parent class

**Attributes:**
- `cfg`: Environment configuration
- `scene`: Scene containing robot and objects
- `num_envs`: Number of parallel environments
- `device`: PyTorch device

**Methods:**

##### `reset()`
Reset the environment.

**Returns:**
- `dict`: Initial observations

##### `step(actions: torch.Tensor)`
Execute one step in the environment.

**Parameters:**
- `actions` (torch.Tensor): Actions to execute, shape (num_envs, action_dim)

**Returns:**
- `observations` (dict): Observations
- `rewards` (torch.Tensor): Rewards, shape (num_envs,)
- `dones` (torch.Tensor): Done flags, shape (num_envs,)
- `infos` (dict): Additional information

#### `rl_arm.envs.ur5e.UR5eEnvCfg`

Configuration dataclass for UR5e environment.

**Attributes:**
- `scene` (UR5eSceneCfg): Scene configuration
- `observations` (ObservationsCfg): Observation configuration
- `rewards` (RewardsCfg): Reward configuration
- `terminations` (TerminationsCfg): Termination configuration
- `events` (EventCfg): Event configuration
- `decimation` (int): Control decimation
- `episode_length_s` (float): Episode length in seconds
- `sim` (SimulationCfg): Simulation configuration

### Franka Research3 Environment

#### `rl_arm.envs.franka_research3.FrankaResearch3Env`

Environment for Franka Research3 robot arm manipulation tasks.

**Constructor:**
```python
FrankaResearch3Env(cfg: FrankaResearch3EnvCfg, render_mode: str | None = None, **kwargs)
```

**Parameters:**
- `cfg` (FrankaResearch3EnvCfg): Environment configuration
- `render_mode` (str, optional): Render mode
- `**kwargs`: Additional arguments

**Methods:** Same as UR5eEnv

#### `rl_arm.envs.franka_research3.FrankaResearch3EnvCfg`

Configuration dataclass for Franka Research3 environment.

**Attributes:** Same structure as UR5eEnvCfg

## Utilities

### Observation Functions

Module: `rl_arm.utils.observations`

#### `ee_position(env, asset_cfg="robot")`
Get end-effector position in world frame.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Position, shape (num_envs, 3)

#### `ee_orientation(env, asset_cfg="robot")`
Get end-effector orientation quaternion in world frame.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Quaternion, shape (num_envs, 4)

#### `ee_velocity(env, asset_cfg="robot")`
Get end-effector linear velocity in world frame.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Velocity, shape (num_envs, 3)

#### `joint_positions_normalized(env, asset_cfg="robot")`
Get normalized joint positions in range [-1, 1].

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Normalized positions, shape (num_envs, num_joints)

#### `target_position_relative(env, target_name="target", asset_cfg="robot")`
Get target position relative to end-effector.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `target_name` (str): Name of target object
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Relative position, shape (num_envs, 3)

### Reward Functions

Module: `rl_arm.utils.rewards`

#### `reaching_reward(env, target_name="target", asset_cfg="robot", std=0.1)`
Dense reaching reward using Gaussian kernel.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `target_name` (str): Name of target object
- `asset_cfg` (str): Name of robot asset
- `std` (float): Standard deviation for Gaussian

**Returns:**
- `torch.Tensor`: Rewards, shape (num_envs,)

#### `reaching_position_l2(env, target_name="target", asset_cfg="robot")`
Negative L2 distance to target.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `target_name` (str): Name of target object
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Rewards, shape (num_envs,)

#### `joint_velocity_penalty(env, asset_cfg="robot", scale=0.01)`
Penalty for large joint velocities.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset
- `scale` (float): Penalty scale factor

**Returns:**
- `torch.Tensor`: Penalties, shape (num_envs,)

#### `action_rate_penalty(env, scale=0.01)`
Penalty for large action changes.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `scale` (float): Penalty scale factor

**Returns:**
- `torch.Tensor`: Penalties, shape (num_envs,)

#### `joint_limits_penalty(env, asset_cfg="robot", threshold=0.95, scale=1.0)`
Penalty for approaching joint limits.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset
- `threshold` (float): Threshold fraction for penalty
- `scale` (float): Penalty scale factor

**Returns:**
- `torch.Tensor`: Penalties, shape (num_envs,)

#### `orientation_alignment_reward(env, target_name="target", asset_cfg="robot", scale=1.0)`
Reward for aligning end-effector orientation with target.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `target_name` (str): Name of target object
- `asset_cfg` (str): Name of robot asset
- `scale` (float): Reward scale factor

**Returns:**
- `torch.Tensor`: Rewards, shape (num_envs,)

### Sensor Functions

Module: `rl_arm.utils.sensors`

#### `compute_jacobian(env, asset_cfg="robot")`
Compute geometric Jacobian of end-effector.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Jacobian, shape (num_envs, 6, num_joints)

#### `compute_forward_kinematics(env, joint_positions, asset_cfg="robot")`
Compute forward kinematics for given joint positions.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `joint_positions` (torch.Tensor): Joint positions
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `tuple[torch.Tensor, torch.Tensor]`: Position and orientation

#### `compute_contact_forces(env, asset_cfg="robot", link_name="ee_link")`
Get contact forces on a specific link.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset
- `link_name` (str): Name of link

**Returns:**
- `torch.Tensor`: Contact forces, shape (num_envs, 3)

#### `check_self_collision(env, asset_cfg="robot")`
Check for self-collisions.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Boolean collision flags, shape (num_envs,)

#### `get_wrench(env, asset_cfg="robot")`
Get wrench (force and torque) at end-effector.

**Parameters:**
- `env` (ManagerBasedRLEnv): Environment instance
- `asset_cfg` (str): Name of robot asset

**Returns:**
- `torch.Tensor`: Wrench [force, torque], shape (num_envs, 6)

## Configuration

### Training Configuration

Module: `rl_arm.configs.train.train_cfg`

#### `PPOTrainCfg`

Configuration for PPO training.

**Attributes:**
- `num_iterations` (int): Number of training iterations
- `num_steps_per_env` (int): Steps per environment per iteration
- `learning_rate` (float): Learning rate
- `gamma` (float): Discount factor
- `gae_lambda` (float): GAE lambda parameter
- `clip_param` (float): PPO clip parameter
- `value_loss_coef` (float): Value loss coefficient
- `entropy_coef` (float): Entropy coefficient
- `max_grad_norm` (float): Max gradient norm for clipping
- `num_layers` (int): Number of network layers
- `hidden_dim` (int): Hidden layer dimension
- `activation` (str): Activation function
- `log_interval` (int): Logging interval
- `save_interval` (int): Save interval
- `device` (str): Device to use
- `seed` (int): Random seed

#### `SACTrainCfg`

Configuration for SAC training.

**Attributes:**
- `num_iterations` (int): Number of training iterations
- `batch_size` (int): Batch size
- `learning_rate` (float): Learning rate
- `gamma` (float): Discount factor
- `tau` (float): Soft update coefficient
- `alpha` (float): Entropy temperature
- `automatic_entropy_tuning` (bool): Auto-tune alpha
- `buffer_size` (int): Replay buffer size
- Additional attributes similar to PPOTrainCfg

#### `TD3TrainCfg`

Configuration for TD3 training.

**Attributes:**
- `num_iterations` (int): Number of training iterations
- `batch_size` (int): Batch size
- `learning_rate` (float): Learning rate
- `gamma` (float): Discount factor
- `tau` (float): Soft update coefficient
- `policy_noise` (float): Policy smoothing noise
- `noise_clip` (float): Noise clipping range
- `policy_delay` (int): Policy update delay
- `buffer_size` (int): Replay buffer size
- `exploration_noise` (float): Exploration noise
- Additional attributes similar to PPOTrainCfg

## Examples

### Creating a Custom Environment

```python
from rl_arm.envs.ur5e import UR5eEnvCfg
from rl_arm.utils import rewards, observations
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils import configclass

@configclass
class MyCustomEnvCfg(UR5eEnvCfg):
    """Custom environment configuration."""
    
    @configclass
    class CustomRewardsCfg:
        reaching = RewTerm(
            func=rewards.reaching_reward,
            weight=10.0,
            params={"target_name": "target", "std": 0.1}
        )
        
        joint_vel = RewTerm(
            func=rewards.joint_velocity_penalty,
            weight=0.01
        )
    
    rewards: CustomRewardsCfg = CustomRewardsCfg()
```

### Using Custom Observations

```python
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from rl_arm.utils import observations

@configclass
class CustomObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=lambda env: env.scene.robot.data.joint_pos)
        ee_pos = ObsTerm(func=observations.ee_position)
        target_rel = ObsTerm(func=observations.target_position_relative)
        
        def __post_init__(self):
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
```

## Type Hints

All functions use Python type hints for better IDE support:

```python
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def my_function(env: ManagerBasedRLEnv, param: float = 1.0) -> torch.Tensor:
    """Function with proper type hints."""
    pass
```
