
# RL Arm: Robot Manipulation Simulation

 [Isaac Sim](https://developer.nvidia.com/isaac-sim) and [MuJoCo](https://mujoco.org/) for robot manipulation simulation.

 [![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.1.0-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-red.svg)](https://releases.ubuntu.com/22.04/)
[![skrl](https://img.shields.io/badge/skrl-1.4.3-yellow.svg)](https://skrl.readthedocs.io/en/latest/)
[![SB3](https://img.shields.io/badge/SB3-2.6.0-green.svg)](https://github.com/DLR-RM/stable-baselines3)

---------------------------------------------------------------------------------------------------

## Goals

- **Short-term:** Implement basic manipulation tasks.
- **Long-term:** Integrate tactile sensing capabilities.

Both in Isaac Sim and MuJoCo. 

## Training

In [`RL_Dog'](https://github.com/pietrodardano/RL_Dog) you can have a look at the main.py script in "Isaac_aliengo" folder.
That approach for training is still valid for this project as well.

However i will follow the more classic and straighforward approach: 
1) Configure and Setup the environment e.g. ur5e_specific_task_cfg.py
2) Config the SKRL algorithm hyperparameters (e.g. skrl_ppo_cfg.yaml)
3) Register the environment using gym.register() 
4) Run training as follows:

```bash
conda activate isaacenv
cd 
cd IsaacLab

python scripts/skrl/train.py --task Task_Name_v0 --headless # --video
```

### Reinforcement Learning Algorithms (via [SKRL](https://skrl.readthedocs.io/))

- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Soft Actor-Critic (SAC)


## Installation

To set up the project, follow these steps:
1. Install IsaacSim and IsaacLab via the official documentation.
2. Clone/fork this repository:
   ```
   git clone https://github.com/pietrodardano/RL_Arm.git
   ```
3. Check that your assets (URDF, config) are installed **locally**, in your IsaacLab folder in isaaclab_assets directory.
4. I am using Miniconda, be sure to change or use the same environment name.
5. Launch the simulation (headless or not) with the scripts that you can find at the beginning of each **main.py** (if present) or as shown above.



## Used Workstation Specs

| Workstation    | CPU                        | GPU                                         | RAM    | OS                   |
|----------------|----------------------------|---------------------------------------------|--------|----------------------|
| **WS 1**       | AMD® Ryzen 9 7950x         | 2x NVIDIA RTX A6000 Ada Gen, 48GB GDDR6, 300W | 192GB  | Ubuntu 22.04.4 LTS   |
| **WS 2**       | Intel Xeon® Gold 6226R     | NVIDIA RTX A6000, 48GB GDDR6, 300W          | 128GB  | Ubuntu 20.04 LTS     |
| **WS 3**       | Intel Xeon® Gold 5415+     | NVIDIA RTX A4000, 14GB GDDR6, 140W          | 128GB  | Ubuntu 20.04 LTS     |
| **WS 4** 📌    | AMD® Ryzen Threadripper 7970x | NVIDIA RTX PRO Blackwell A6000, 96GB GDDR7 | 128GB  | Ubuntu 22.04.4 LTS   |

### System Configs  --> Nvidia & CUDA:
- **Driver Version**: 570.124.06  
- **CUDA Version**: 12.8  
- **For Nvidia Blackwell**: Driver 570.133.20 (server-open) | CUDA Version 12.8  

## Extra

**NVIDIA's [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)**: allows for parallel simulation of multiple environments necessary for training our models. Refer to the [Orbit](https://isaac-orbit.github.io/) and [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) pages for more information. <br>

Please note that IsaacLab contains many OpenAI Gym and Gymnasium features. It is common to find attributes, methods and classes related to them. <br>
