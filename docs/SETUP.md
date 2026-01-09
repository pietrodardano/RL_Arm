# Setup Guide

This guide provides detailed instructions for setting up the RL_Arm project.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with compute capability 7.0+ (RTX 2060 or better recommended)
- **RAM**: 32GB+ recommended for large-scale parallel simulation
- **Storage**: 50GB+ free space for Isaac Sim and dependencies

### Software
- **OS**: Ubuntu 20.04/22.04 or Windows 10/11
- **GPU Driver**: NVIDIA driver 525.60.11 or later
- **Python**: 3.10 or 3.11

## Installation Steps

### 1. Install NVIDIA Isaac Sim

Isaac Sim is required as the base simulation platform.

#### Option A: Install via Omniverse Launcher (Recommended)

1. Download and install [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/)
2. In the Launcher, go to the "Exchange" tab
3. Search for "Isaac Sim" and install version 5.1.0 or compatible
4. Launch Isaac Sim once to verify installation

#### Option B: Install Standalone

Follow the instructions at: https://docs.isaacsim.omniverse.nvidia.com/latest/index.html

### 2. Install IsaacLab

IsaacLab provides the framework for robot learning environments.

```bash
# Clone IsaacLab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run the installation script
./isaaclab.sh --install

# Verify installation
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

### 3. Install RL_Arm

```bash
# Clone RL_Arm repository
cd ~
git clone https://github.com/pietrodardano/RL_Arm.git
cd RL_Arm

# Activate IsaacLab environment
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh

# Install RL_Arm in development mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test import
python -c "import rl_arm; print('RL_Arm installed successfully!')"

# Test environment configuration
python -c "from rl_arm.envs.ur5e import UR5eEnvCfg; print('UR5e config loaded!')"
python -c "from rl_arm.envs.franka_research3 import FrankaResearch3EnvCfg; print('FR3 config loaded!')"
```

## Environment Setup

### Activate Environment

Before running any scripts, activate the IsaacLab environment:

```bash
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
cd /path/to/RL_Arm
```

### Set Environment Variables (Optional)

```bash
# For better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# For debugging
export ISAACLAB_VERBOSE=1
```

## Troubleshooting

### Issue: ImportError for omni.isaac modules

**Solution**: Ensure you've activated the IsaacLab environment:
```bash
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
```

### Issue: CUDA out of memory

**Solution**: Reduce the number of parallel environments:
```bash
python rl_arm/scripts/train_ur5e.py --num_envs 1024  # Instead of 4096
```

### Issue: Graphics/rendering errors

**Solution**: Run in headless mode:
```bash
python rl_arm/scripts/train_ur5e.py --headless
```

### Issue: Slow simulation

**Solution**: 
1. Ensure GPU drivers are up to date
2. Reduce simulation complexity
3. Check GPU utilization with `nvidia-smi`

## Development Setup

For development, install additional tools:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Next Steps

- Read [USAGE.md](USAGE.md) for usage examples
- Check [API.md](API.md) for API documentation
- See the main [README.md](../README.md) for quick start guide
