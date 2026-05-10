# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS 2 (Jazzy) workspace for training and evaluating a **Spiking Deep Deterministic Policy Gradient (SDDPG)** agent for mapless robot navigation. It is a migration from a ROS 1 + Gazebo Classic project to ROS 2 + Gazebo Harmonic.

**Robot:** TurtleBot3 Burger in Gazebo Harmonic
**Task:** Reach goals while avoiding obstacles using LiDAR (18 beams), goal direction/distance, and odometry
**State space:** 22 dimensions | **Action space:** 2 continuous (left/right wheel speeds)

## Setup

Source these two lines at the start of every terminal session (or add to `~/.bashrc`):

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Build & Run Commands

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Training

Requires two terminals. Terminal 1 (Gazebo + bridge):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch sddpg_navigation training.launch.py

# Headless (no GUI, faster — useful for remote/SSH sessions)
ros2 launch sddpg_navigation training.launch.py headless:=true
```

Terminal 2 (training script, after Gazebo is up):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# Standard DDPG (TD3 with GRU actor)
ros2 run sddpg_navigation train_ddpg

# Spiking DDPG
ros2 run sddpg_navigation train_sddpg
```

### Evaluation (simulation)

Terminal 1 (Gazebo + bridge):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch sddpg_navigation evaluation.launch.py

# Headless (no GUI)
ros2 launch sddpg_navigation evaluation.launch.py headless:=true
```

Terminal 2 (evaluation script):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# Evaluate DDPG (loads from evaluation/saved_model/ by default)
ros2 run sddpg_navigation eval_ddpg

# Evaluate Spiking DDPG (loads from evaluation/saved_model/ by default)
ros2 run sddpg_navigation eval_sddpg

# Evaluate a specific SDDPG checkpoint from training
ros2 run sddpg_navigation eval_sddpg \
  --model_name SNN_R1 \
  --checkpoint 0 \
  --save_dir src/sddpg_navigation/sddpg_navigation/save_sddpg_weights/
```

### Real-world evaluation

Run directly (no Gazebo needed):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# DDPG on real robot
python3 src/sddpg_navigation/sddpg_navigation/evaluation/eval_real_world/run_ddpg_eval_rw.py

# Spiking DDPG on real robot (requires Loihi hardware)
python3 src/sddpg_navigation/sddpg_navigation/evaluation/eval_real_world/run_sddpg_loihi_eval_rw.py
```

### Manual environment testing

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run sddpg_navigation environment
```

## Testing & Linting

```bash
# Run all tests
colcon test --packages-select sddpg_navigation
colcon test-result --verbose

# Run individual test types
python3 -m pytest src/sddpg_navigation/test/test_flake8.py -v
python3 -m pytest src/sddpg_navigation/test/test_pep257.py -v
python3 -m pytest src/sddpg_navigation/test/test_copyright.py -v
```

Tests only cover linting (flake8, pep257, copyright). No functional unit tests exist yet.

## Architecture

### Data Flow

```
Gazebo Harmonic (Physics Simulation)
    ↕ ros_gz_bridge (bridge.yaml)
ROS 2 Topics/Services:
    /scan (LaserScan, GZ→ROS)
    /odom (Odometry, GZ→ROS)
    /cmd_vel (Twist, ROS→GZ)
    /world/default/control (pause/unpause, ROS→GZ service)
    /world/default/set_pose (teleport entities, ROS→GZ service)
    ↕ rclpy
GazeboEnvironment Node (environment.py)
    ↕ Python function calls
Agent (DDPGAgent or SDDPGAgent)
    ↕ PyTorch
Neural Networks (actor + critic)
```

### Key Files

| File | Role |
|------|------|
| `src/sddpg_navigation/sddpg_navigation/environment.py` | `GazeboEnvironment(Node)` — gym-like RL environment; provides `step()`, `reset()`, `set_new_environment()` |
| `src/sddpg_navigation/sddpg_navigation/utility.py` | Training environment generation (`gen_rand_list_env1/2/3/4`), action decoding, state normalization |
| `src/sddpg_navigation/sddpg_navigation/training/train_ddpg/train_ddpg.py` | Main DDPG training loop |
| `src/sddpg_navigation/sddpg_navigation/training/train_ddpg/ddpg_agent.py` | DDPG agent with experience replay buffer |
| `src/sddpg_navigation/sddpg_navigation/training/train_ddpg/ddpg_networks.py` | PyTorch `ActorNet` (256×3 FC) and `CriticNet` (512×3 FC) |
| `src/sddpg_navigation/sddpg_navigation/training/train_spiking_ddpg/train_sddpg.py` | Main SDDPG training loop |
| `src/sddpg_navigation/sddpg_navigation/training/train_spiking_ddpg/sddpg_agent.py` | Spiking DDPG agent |
| `src/sddpg_navigation/sddpg_navigation/training/train_spiking_ddpg/sddpg_networks.py` | `ActorNetSpiking` — LIF neurons with STBP; shares `CriticNet` |
| `src/sddpg_navigation/sddpg_navigation/evaluation/eval_random_simulation/run_ddpg_eval.py` | DDPG simulation evaluation |
| `src/sddpg_navigation/sddpg_navigation/evaluation/eval_random_simulation/run_sddpg_eval.py` | SDDPG simulation evaluation |
| `src/sddpg_navigation/sddpg_navigation/evaluation/eval_real_world/run_ddpg_eval_rw.py` | DDPG real-robot evaluation |
| `src/sddpg_navigation/sddpg_navigation/evaluation/eval_real_world/run_sddpg_loihi_eval_rw.py` | SDDPG real-robot evaluation (Loihi) |
| `src/sddpg_navigation/sddpg_navigation/evaluation/result_analyze/generate_results.py` | Result analysis and plotting |
| `src/sddpg_navigation/launch/training.launch.py` | Starts Gazebo + bridge for training |
| `src/sddpg_navigation/launch/evaluation.launch.py` | Starts Gazebo + bridge for evaluation |
| `src/sddpg_navigation/launch/bridge.yaml` | ROS↔Gazebo topic/service bridge configuration |
| `src/sddpg_navigation/worlds/training_worlds.world` | 4 training environments (SDF 1.9) |
| `src/sddpg_navigation/worlds/evaluation_world.world` | Evaluation environment |
| `src/sddpg_navigation/sddpg_navigation/random_positions/` | Pre-generated random init/goal positions (pickle, 5 sets × 4 envs × 100 episodes) |

### Agent Variants

- **DDPG**: Standard actor-critic with experience replay and soft target updates. Uses GPU (CUDA).
- **SDDPG**: Spiking actor (LIF neurons, STBP training) + standard critic. Designed for deployment on Intel Loihi neuromorphic hardware.

### Reward Structure

| Event | Reward |
|-------|--------|
| Goal reached | +30 |
| Collision (LiDAR < threshold) | −20 |
| Per step | 15 × Δdistance_to_goal |

### Training Environments

4 environments with different obstacle layouts defined in `utility.py` (`gen_rand_list_env1–4`). Each episode randomly places the robot and goal using pre-generated positions from `random_positions/`.

## Migration Notes

This project is an active ROS 1 → ROS 2 port. Key API changes from the original:

- `pause_physics` / `unpause_physics` → `/world/default/control` (WorldControl service)
- `SetModelState` → `/world/default/set_pose` (Pose service via ros_gz_interfaces)
- Custom `simplescan` topic → standard `/scan` (LaserScan)
- Gazebo Classic SDF → Gazebo Harmonic SDF 1.9 (inline lights, no `model://sun`)

## Package Structure

```
src/sddpg_navigation/          # ROS 2 package (ament_python)
├── launch/                    # Launch files and bridge config
│   ├── training.launch.py
│   ├── evaluation.launch.py
│   └── bridge.yaml
├── models/turtlebot3_burger/  # Robot SDF model
├── worlds/                    # Gazebo world files
├── sddpg_navigation/          # Python package
│   ├── environment.py
│   ├── utility.py
│   ├── training/
│   │   ├── train_ddpg/
│   │   └── train_spiking_ddpg/
│   ├── evaluation/
│   │   ├── eval_random_simulation/  # Simulation evaluation scripts
│   │   ├── eval_real_world/         # Real-robot evaluation scripts
│   │   ├── record_data/             # Saved model weights and recorded trajectories
│   │   └── result_analyze/          # Result analysis and plotting
│   └── random_positions/      # Pickle files with pre-generated positions
└── test/                      # Linting tests only
```
