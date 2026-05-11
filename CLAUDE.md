# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 (Jazzy) workspace for mapless robot navigation using reinforcement learning. Three agent variants are implemented:

- **TD3 (LiDAR)** — Twin Delayed DDPG with GRU actor and sequential experience replay. Primary training target.
- **DVS (Event Camera)** — TD3 with CNN+GRU actor using synthetic event camera input. LiDAR critic retained.
- **SDDPG (Spiking)** — Legacy spiking actor (LIF + STBP) for Intel Loihi deployment.

**Robot:** TurtleBot3 Burger in Gazebo Harmonic
**Task:** Reach goal while avoiding obstacles
**State (LiDAR):** 22 dim — `[goal_dir, goal_dis, odom_lin, odom_ang, lidar×18]`
**State (DVS):** event frame `(2, 64, 64)` float32 (ON/OFF channels) + goal `state[:2]`
**Action:** 2 continuous (left/right wheel speeds, decoded to m/s)

## Setup

Source at the start of every terminal (or add to `~/.bashrc`):

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

**Note:** After any code change, `colcon build --symlink-install` is required because Python files in `build/` are copies (not symlinks for this package type). Without rebuild, old code runs.

## Training — TD3 LiDAR

Terminal 1 (Gazebo + bridge):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch sddpg_navigation training.launch.py headless:=true

# With GUI (local machine with GPU only — server shows empty Gazebo)
ros2 launch sddpg_navigation training.launch.py
```

Terminal 2 (after Gazebo is up):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run sddpg_navigation train_ddpg

# Start from a specific environment (0=env1, 1=env2, 2=env3, 3=env4)
ros2 run sddpg_navigation train_ddpg --start_env 1
```

## Training — DVS (Event Camera)

Uses `training_dynamic.launch.py` which includes: Gazebo + bridge + event_camera node + obstacle mover.

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch sddpg_navigation training_dynamic.launch.py headless:=true
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run sddpg_navigation train_dvs_ddpg

# Start from specific environment
ros2 run sddpg_navigation train_dvs_ddpg --start_env 1
```

### Viewing DVS output (optional, separate terminal)

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run rqt_image_view rqt_image_view /camera/events     # ON=white, OFF=black, neutral=grey
ros2 run rqt_image_view rqt_image_view /camera/image_raw  # raw camera before processing
```

## Training — Dynamic Obstacles (LiDAR)

```bash
# Terminal 1
ros2 launch sddpg_navigation training_dynamic.launch.py headless:=true

# Terminal 2
ros2 run sddpg_navigation train_ddpg
```

## Evaluation — Simulation

Terminal 1 (Gazebo + bridge):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch sddpg_navigation evaluation.launch.py headless:=true
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run sddpg_navigation eval_ddpg
ros2 run sddpg_navigation eval_sddpg
```

## Evaluation — Dynamic Obstacles

```bash
# Terminal 1 (Gazebo + bridge + obstacle mover + event_camera)
ros2 launch sddpg_navigation evaluation_dynamic.launch.py headless:=true

# Terminal 2
ros2 run sddpg_navigation eval_ddpg
```

## Evaluation — Real World

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
python3 src/sddpg_navigation/sddpg_navigation/evaluation/eval_real_world/run_ddpg_eval_rw.py
python3 src/sddpg_navigation/sddpg_navigation/evaluation/eval_real_world/run_sddpg_loihi_eval_rw.py
```

## Architecture

### TD3 LiDAR Agent

```
state (22 dim) ──► ActorNet: FC1→FC2→GRU→FC3 ──► action (2 dim)
                   (256, 256, 256, batch_first=True)
                   last_action concatenated at GRU input

state (22 dim) ──► CriticNet×2: FC1→FC2→FC3→FC4 ──► Q value
action (2 dim) ──┘  (512, 512, 512)
```

TD3 improvements: twin critics (min Q), delayed policy update (`policy_delay=2`), target policy smoothing, gradient clipping (`max_norm=1.0`).

Sequential buffer: stores sequences of `seq_len=10` steps with initial GRU hidden state and last_action. Memory entries ~640 KB each.

### DVS Agent

```
event_frame (2,64,64) ──► CNN: Conv→Pool→Conv→Pool→FC(8192→256) ──┐
goal = state[:2]       ──► FC(2→64) ──────────────────────────────┼──► GRU(322→256) ──► FC→Sigmoid ──► action
last_action (2 dim)    ─────────────────────────────────────────────┘

state (22 dim) ──► CriticNet×2 (LiDAR, unchanged) ──► Q value
action (2 dim) ──┘
```

Event decoding in `train_DVS_ddpg.py`: `on = raw > 200`, `off = raw < 50` → float32 (2, 64, 64).

### State Structure

```
state[0]   = goal_dir         (direction to goal)
state[1]   = goal_dis         (normalized distance to goal)
state[2]   = odom_linear      (linear velocity)
state[3]   = odom_angular     (angular velocity)
state[4:22] = lidar×18        (front hemisphere, left→right)
```

Goal for DVS actor: `state[:2]` (NOT `state[-2:]`).

### Reward Structure

| Event | Reward |
|-------|--------|
| Goal reached | +30 |
| Collision (LiDAR < threshold) | −20 |
| Per step | 15 × Δdistance_to_goal |

### Training Environments

4 environments with increasing difficulty. Positions pre-generated in `random_positions/` (100 episodes per env). `episode_num=(100, 200, 300, 400)` by default.

### Event Camera

- Robot model: `models/turtlebot3_burger/model.sdf` — camera sensor on `camera_link`
- Bridge: `/camera/image_raw` GZ→ROS in `bridge.yaml`
- Processing: `event_camera.py` — median blur → frame diff → threshold (0.003) → encode ON=255/OFF=0/neutral=128
- Published: `/camera/events` (mono8, 64×64)

### Dynamic Obstacles

- `dynamic_obstacles.py` — moves obstacles at 20 Hz via `/world/default/set_pose`
- Training world: env1 (4 obstacles) + env4 (3 obstacles)
- Evaluation world: 6 obstacles spread across 22×22 m arena
- Parameter `mode`: `training` (default) or `evaluation` — set in launch file

## Key Files

| File | Role |
|------|------|
| `environment.py` | `GazeboEnvironment(Node)` — `step()`, `reset()`, `set_new_environment()` |
| `utility.py` | Env generation, action decoding, state normalization |
| `training/train_ddpg/train_ddpg.py` | TD3 LiDAR training loop |
| `training/train_ddpg/ddpg_agent.py` | TD3 agent — sequential buffer, hidden state, last_action |
| `training/train_ddpg/ddpg_networks.py` | `ActorNet` (FC+GRU) + `CriticNet` |
| `training/train_DVS_ddpg/train_DVS_ddpg.py` | DVS training loop — `EventSubscriber` node |
| `training/train_DVS_ddpg/ddpg_DVS_agent.py` | DVS agent — events buffer, goal=state[:2] |
| `training/train_DVS_ddpg/ddpg_DVS_networks.py` | `ActorNet` (CNN+GRU) + `CriticNet` |
| `event_camera.py` | Publishes synthetic DVS events from `/camera/image_raw` |
| `dynamic_obstacles.py` | Moves dynamic obstacles in Gazebo |
| `launch/training.launch.py` | Gazebo + bridge (static world) |
| `launch/training_dynamic.launch.py` | Gazebo + bridge + event_camera + obstacle_mover |
| `launch/evaluation.launch.py` | Evaluation (static world) |
| `launch/evaluation_dynamic.launch.py` | Evaluation (dynamic world + event_camera) |
| `launch/bridge.yaml` | ROS↔Gazebo topic/service bridge config |
| `worlds/training_worlds.world` | 4 static training environments |
| `worlds/training_dynamic_world.world` | 4 environments + dynamic obstacles (env1, env4) |
| `worlds/evaluation_world.world` | Static evaluation environment |
| `worlds/evaluation_dynamic_world.world` | Evaluation + 6 dynamic obstacles |
| `models/turtlebot3_burger/` | Robot SDF (LiDAR + camera sensor) |
| `models/textures/` | `vertical_stripes.png` used by dynamic world walls |
| `random_positions/` | Pre-generated start/goal positions (pickle) |

## Package Structure

```
src/sddpg_navigation/
├── launch/
│   ├── training.launch.py
│   ├── training_dynamic.launch.py      ← event_camera + obstacle_mover
│   ├── evaluation.launch.py
│   ├── evaluation_dynamic.launch.py    ← event_camera + obstacle_mover
│   └── bridge.yaml
├── models/
│   ├── turtlebot3_burger/
│   └── textures/                       ← vertical_stripes.png, model.config
├── worlds/
│   ├── training_worlds.world
│   ├── training_dynamic_world.world
│   ├── evaluation_world.world
│   └── evaluation_dynamic_world.world
└── sddpg_navigation/
    ├── environment.py
    ├── utility.py
    ├── event_camera.py
    ├── dynamic_obstacles.py
    ├── training/
    │   ├── train_ddpg/                 ← TD3 LiDAR
    │   ├── train_DVS_ddpg/             ← TD3 DVS
    │   └── train_spiking_ddpg/         ← SDDPG (legacy)
    ├── evaluation/
    │   ├── eval_random_simulation/
    │   ├── eval_real_world/
    │   └── result_analyze/
    └── random_positions/
```

## Known Issues

- Gazebo GUI shows empty world on server without local GPU — use `headless:=true`
- `models/textures/` must not contain broken symlinks — breaks Gazebo model loading
- After `colcon build` without `--symlink-install`, Python changes require full rebuild
