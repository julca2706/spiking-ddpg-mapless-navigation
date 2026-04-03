# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS 2 (Jazzy) workspace for training and evaluating a **Spiking Deep Deterministic Policy Gradient (SDDPG)** agent for mapless robot navigation. It is a migration from a ROS 1 + Gazebo Classic project to ROS 2 + Gazebo Harmonic.

**Robot:** TurtleBot3 Burger in Gazebo Harmonic
**Task:** Reach goals while avoiding obstacles using LiDAR (18 beams), goal direction/distance, and odometry
**State space:** 22 dimensions | **Action space:** 2 continuous (left/right wheel speeds)

## Build & Run Commands

```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Build workspace
cd /home/juleczka123/sddpg_ws
colcon build --symlink-install

# Source workspace
source install/setup.bash

# Launch Gazebo + ros_gz_bridge (required before training)
ros2 launch sddpg_navigation training.launch.py

# Run DDPG training (in a separate terminal, after sourcing)
ros2 run sddpg_navigation train_ddpg

# Run SDDPG training
ros2 run sddpg_navigation train_sddpg

# Run environment node (manual testing)
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
| `sddpg_navigation/environment.py` | `GazeboEnvironment(Node)` — gym-like RL environment; provides `step()`, `reset()`, `set_new_environment()` |
| `sddpg_navigation/utility.py` | Training environment generation (`gen_rand_list_env1/2/3/4`), action decoding, state normalization |
| `training/train_ddpg/train_ddpg.py` | Main DDPG training loop |
| `training/train_ddpg/ddpg_agent.py` | DDPG agent with experience replay buffer |
| `training/train_ddpg/ddpg_networks.py` | PyTorch `ActorNet` (256×3 FC) and `CriticNet` (512×3 FC) |
| `training/train_spiking_ddpg/train_sddpg.py` | Main SDDPG training loop |
| `training/train_spiking_ddpg/sddpg_agent.py` | Spiking DDPG agent |
| `training/train_spiking_ddpg/sddpg_networks.py` | `ActorNetSpiking` — LIF neurons with STBP; shares `CriticNet` |
| `launch/training.launch.py` | Starts Gazebo + bridge |
| `launch/bridge.yaml` | ROS↔Gazebo topic/service bridge configuration |
| `worlds/training_worlds.world` | 4 training environments (SDF 1.9) |
| `worlds/evaluation_world.world` | Evaluation environment |
| `sddpg_navigation/random_positions/` | Pre-generated random init/goal positions (pickle, 5 sets × 4 envs × 100 episodes) |

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

This project is an active ROS 1 → ROS 2 port. Key API changes from the original (`src/spiking-ddpg-mapless-navigation/`):

- `pause_physics` / `unpause_physics` → `/world/default/control` (WorldControl service)
- `SetModelState` → `/world/default/set_pose` (Pose service via ros_gz_interfaces)
- Custom `simplescan` topic → standard `/scan` (LaserScan)
- Gazebo Classic SDF → Gazebo Harmonic SDF 1.9 (inline lights, no `model://sun`)

See `src/notes.md` for detailed migration documentation (in Polish).

## Package Structure

```
src/sddpg_navigation/          # ROS 2 package (ament_python)
├── launch/                    # Launch files and bridge config
├── models/turtlebot3_burger/  # Robot SDF model
├── worlds/                    # Gazebo world files
├── sddpg_navigation/          # Python package
│   ├── environment.py
│   ├── utility.py
│   ├── training/
│   │   ├── train_ddpg/
│   │   └── train_spiking_ddpg/
│   ├── evaluation/            # Partially migrated evaluation scripts
│   │   ├── eval_random_simulation/
│   │   ├── eval_random_simulation_loihi/
│   │   ├── eval_real_world/
│   │   └── loihi_network/
│   └── random_positions/      # Pickle files with pre-generated positions
└── test/                      # Linting tests only
src/spiking-ddpg-mapless-navigation/  # Original ROS 1 project (reference only)
```
