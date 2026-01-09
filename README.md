# RL_Arm

Reinforcement Learning for Robot Arm Manipulators using IsaacLab.

This repository provides environments and utilities for training RL agents on robot arm manipulators, specifically:
- **UR5e** (Universal Robots 5e)
- **Franka Research3** (FR3)

Built on [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab), leveraging GPU-accelerated physics simulation for fast and efficient training.

## 🚀 Features

- **Multi-Robot Support**: Pre-configured environments for UR5e and Franka Research3
- **Modular Architecture**: Easy-to-extend structure for custom tasks
- **GPU Acceleration**: Leverages IsaacSim for parallel simulation across thousands of environments
- **Custom Utilities**: Observation, reward, and sensor helper functions
- **Multiple RL Algorithms**: Support for PPO, SAC, and TD3 (templates provided)
- **Comprehensive Documentation**: Well-documented codebase with examples

## 📁 Repository Structure

```
RL_Arm/
├── rl_arm/                          # Main package
│   ├── envs/                        # Environment definitions
│   │   ├── ur5e/                    # UR5e robot environments
│   │   │   ├── __init__.py
│   │   │   ├── ur5e_env.py          # Environment implementation
│   │   │   └── ur5e_env_cfg.py      # Environment configuration
│   │   └── franka_research3/        # Franka Research3 environments
│   │       ├── __init__.py
│   │       ├── franka_research3_env.py
│   │       └── franka_research3_env_cfg.py
│   ├── agents/                      # RL agent implementations
│   │   └── __init__.py
│   ├── configs/                     # Configuration files
│   │   ├── env/                     # Environment configs
│   │   ├── agent/                   # Agent configs
│   │   └── train/                   # Training configs
│   │       └── train_cfg.py         # Training configurations
│   ├── scripts/                     # Training and evaluation scripts
│   │   ├── train_ur5e.py           # UR5e training script
│   │   ├── train_franka_research3.py # FR3 training script
│   │   └── evaluate.py             # Evaluation script
│   ├── utils/                       # Utility modules
│   │   ├── observations.py         # Custom observation functions
│   │   ├── rewards.py              # Custom reward functions
│   │   └── sensors.py              # Sensor utilities
│   └── __init__.py
├── docs/                            # Documentation
│   ├── SETUP.md                    # Setup instructions
│   ├── USAGE.md                    # Usage guide
│   └── API.md                      # API documentation
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠️ Installation

### Prerequisites

1. **NVIDIA Isaac Sim**: Install Isaac Sim 5.1.0 or compatible version
   - Follow instructions at: https://docs.isaacsim.omniverse.nvidia.com/latest/index.html

2. **IsaacLab**: Clone and install IsaacLab
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   # Follow IsaacLab installation instructions
   ./isaaclab.sh --install
   ```

### Install RL_Arm

1. Clone this repository:
   ```bash
   git clone https://github.com/pietrodardano/RL_Arm.git
   cd RL_Arm
   ```

2. Install in development mode:
   ```bash
   # Activate IsaacLab environment first
   source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
   
   # Install this package
   pip install -e .
   ```

## 🎯 Quick Start

### Training an Agent

#### UR5e Environment
```bash
python rl_arm/scripts/train_ur5e.py \
    --task UR5e-Reach \
    --num_envs 4096 \
    --algo ppo \
    --num_iterations 1000
```

#### Franka Research3 Environment
```bash
python rl_arm/scripts/train_franka_research3.py \
    --task FrankaResearch3-Reach \
    --num_envs 4096 \
    --algo ppo \
    --num_iterations 1000
```

### Evaluating a Trained Policy

```bash
python rl_arm/scripts/evaluate.py \
    --task UR5e-Reach \
    --checkpoint logs/ur5e_reach/checkpoint.pt \
    --num_episodes 100
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[SETUP.md](docs/SETUP.md)**: Complete setup instructions
- **[USAGE.md](docs/USAGE.md)**: Usage guide with examples
- **[API.md](docs/API.md)**: API reference

## 🔧 Customization

### Creating Custom Tasks

1. **Define a new environment configuration** in `rl_arm/envs/<robot>/`
2. **Customize observations**: Use or extend functions in `rl_arm/utils/observations.py`
3. **Define reward functions**: Use or extend functions in `rl_arm/utils/rewards.py`
4. **Create a training script**: See examples in `rl_arm/scripts/`

### Example: Custom Reward Function

```python
from rl_arm.utils.rewards import reaching_reward, joint_velocity_penalty

@configclass
class CustomRewardsCfg:
    """Custom reward configuration."""
    
    reaching = RewTerm(
        func=reaching_reward,
        weight=1.0,
        params={"target_name": "target", "std": 0.1}
    )
    
    joint_vel = RewTerm(
        func=joint_velocity_penalty,
        weight=0.01
    )
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- Universal Robots and Franka Emika for robot models

## 📮 Contact

Pietro Dardano - [GitHub](https://github.com/pietrodardano)

## 🔗 Useful Links

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [UR5e Documentation](https://www.universal-robots.com/products/ur5-robot/)
- [Franka Research 3 Documentation](https://franka.de/research)
