# Code for "Learning Quadrupedal Locomotion on Uneven Terrain"

## Prerequisite
### For building C++ 
- Mujoco 2.3.5 installed at `~/.mujoco/mujoco235/`. (You may change line 26 and 27 of CMakeLists.txt according to your installation of Mujoco).
- glfw3
- X11
- Eigen

### For Python environment
- python=3.7
- numpy=1.22.4
- pytorch=1.8.1
- cudatoolkit=11.1
- scipy>=1.5.0
- tensorboard>=2.2.1
- gym
- omegaconf
- hydra-core>=1.1
- rl_games=1.4.0

## Build
`mkdir build && cd build`
`cmake ..`
`make`

## Run
- To test the environment, enter `parallel_sim` directory under `build` and run `./test_a1_env`. It will create one simulated environment and run with a trained policy model (which was converted from trained .pt model).
- To train the model, enter `parallel_sim` directory under `build` and run `python3 train.py task=MujocoA1 headless=True`. You may specify `max_iteration=XXX` to set maximum training epoches (default to 1500).
- To test the trained model under python environment, enter `parallel_sim` directory under `build` and run `python3 train.py task=MujocoA1 headless=False test=True`.

## File description
```
variable_impedance_RL
├── cmake/
├── parallel_sim/
├── converted_models/
├── python/
│   ├── cfg/
│   └── unitree_a1/
├── CMakeLists.txt
└── Readme.md
```
- `parallel_sim/`: Contains C++ source files for implementation of parallel simulation environment and python interface.
- `converted_models`: Contains two binary model file (converted from .pt model, can be read by  `parallel_sim/eigen_model_from_file.hpp`).
- `python/`: Contains a python wrapper for the parallel simulation environment(`mujoco_a1.py`).
- `python/cfg`: Contains configuration of the environment and hyperparameters for RL algorithm.
- `python/unitrell_a1/`: Contains mujoco model file for Unitree A1 quadrupedal Robot and terrain. The mujoco model for Unitree A1 quadrupedal Robot is copied from [!mujoco_menagerie](https://github.com/deepmind/mujoco_menagerie.git)

