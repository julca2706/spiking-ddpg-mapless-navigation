# Mapless Robot Navigation with Reinforcement Learning

This package implements reinforcement learning agents for mapless autonomous navigation using a TurtleBot3 Burger in Gazebo Harmonic (ROS 2 Jazzy).
The robot must reach goal positions while avoiding static and dynamic obstacles, without access to a pre-built map.

Three agent variants are implemented and compared:

* **TD3 with LiDAR** — Twin Delayed DDPG with GRU actor and sequential experience replay. Primary training target.
* **DVS with Event Camera** — TD3 with CNN+GRU actor using synthetic event camera input. LiDAR critic retained.
* **Spiking DDPG (SDDPG)** — Legacy LIF-neuron actor trained with STBP for Intel Loihi deployment.

This work extends the original SDDPG framework by Tang et al. ([IROS 2020](https://ieeexplore.ieee.org/abstract/document/9340948)) to ROS 2 / Gazebo Harmonic and adds the TD3 and DVS variants.

## Citation

```bibtex
@inproceedings{tang2020reinforcement,
  author    = {Tang, Guangzhi and Kumar, Neelesh and Michmizos, Konstantinos P.},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title     = {Reinforcement co-Learning of Deep and Spiking Neural Networks for Energy-Efficient Mapless Navigation with Neuromorphic Hardware},
  year      = {2020},
  pages     = {6090--6097},
  doi       = {10.1109/IROS45743.2020.9340948}
}
```

## Software Installation

#### 1. System Requirements

* Ubuntu 24.04 LTS
* Python 3.12
* ROS 2 Jazzy + Gazebo Harmonic
* PyTorch >= 2.0 (CUDA optional but recommended for training)

#### 2. ROS 2 and Gazebo

```bash
sudo apt install ros-jazzy-desktop ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-interfaces \
                 ros-jazzy-ros-gz-sim ros-jazzy-cv-bridge \
                 ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-gazebo \
                 gz-harmonic python3-gz-msgs10 python3-gz-transport13 \
                 python3-colcon-common-extensions
```

#### 3. Python Packages

```bash
pip install -r requirements.txt
```

#### 4. Build the Workspace

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Add both `source` lines to `~/.bashrc` to avoid repeating them in every terminal.

After any code change, rebuild with `colcon build --symlink-install` — Python files in `build/` are copies, not symlinks.

## State and Action Spaces

The LiDAR state vector is 22-dimensional:

```
state[0]     goal_dir      — angle to goal relative to robot heading
state[1]     goal_dis      — normalised distance to goal
state[2]     odom_linear   — linear velocity
state[3]     odom_angular  — angular velocity
state[4:22]  lidar × 18    — front hemisphere, left to right
```

The DVS actor receives `state[:2]` (goal direction and distance) together with the event frame and previous action.

The action space is 2-dimensional (left-wheel and right-wheel speeds in `[0, 1]`), decoded to m/s at execution time.

## Network Architectures

**TD3 Actor (GRU)**
```
state (22) + last_action (2)  ──► FC(256) → ReLU → FC(256) → ReLU → GRU(256) → FC(2) → Sigmoid ──► action (2)
```

**TD3 Critic (×2)**
```
state (22) ──► FC(512) → ReLU ──┐
                                 ├──► FC(512) → ReLU → FC(512) → ReLU → FC(1) ──► Q
action (2)  ─────────────────────┘
```

**DVS Actor (CNN + GRU)**
```
events (2,64,64) ──► Conv(16)→Pool→Conv(32)→Pool→FC(8192→256) ──┐
goal (2)         ──► FC(64) ──────────────────────────────────────┼──► GRU(322→256) ──► FC(2) → Sigmoid ──► action (2)
last_action (2)  ────────────────────────────────────────────────┘
```

TD3 improvements applied to all non-legacy agents: twin critics (min-Q), delayed policy update (`policy_delay=2`), target policy smoothing, gradient clipping.
Experience is stored and replayed as sequences of length 10 to train the recurrent actor coherently.

## Training

Each training run requires two terminals. Start Gazebo first, then the agent.

#### 1. TD3 LiDAR — static world

```bash
# Terminal 1
ros2 launch sddpg_navigation training.launch.py headless:=true

# Terminal 2
ros2 run sddpg_navigation train_td3

# Start from a specific environment (0–3)
ros2 run sddpg_navigation train_td3 --start_env 1
```

Weights saved to `save_td3_weights/`.

#### 2. TD3 LiDAR — dynamic obstacles

```bash
# Terminal 1
ros2 launch sddpg_navigation training_dynamic_lidar.launch.py headless:=true

# Terminal 2
ros2 run sddpg_navigation train_td3
```

#### 3. DVS Event Camera

The DVS launch file also starts the `event_camera` node and the obstacle mover.

```bash
# Terminal 1
ros2 launch sddpg_navigation training_dynamic.launch.py headless:=true

# Terminal 2
ros2 run sddpg_navigation train_dvs_ddpg

# Start from a specific environment (0–3)
ros2 run sddpg_navigation train_dvs_ddpg --start_env 1
```

Weights saved to `save_dvs_weights/` as `{run_name}_dvs_actor_s{step}.pt`.

To inspect event camera output while training (optional, separate terminal):

```bash
ros2 run rqt_image_view rqt_image_view /camera/events    # ON=white, OFF=black, neutral=grey
ros2 run rqt_image_view rqt_image_view /camera/image_raw # raw RGB frame
```

#### 4. Pure DDPG Baseline

Simple DDPG without TD3 improvements — FC actor (no GRU), single critic, standard replay buffer.

```bash
# Terminal 1
ros2 launch sddpg_navigation training.launch.py headless:=true

# Terminal 2
ros2 run sddpg_navigation train_pure_ddpg
```

Weights saved to `save_pure_ddpg_weights/`.

## Evaluation

Evaluation runs 200 fixed start/goal pairs from `evaluation/eval_random_simulation/eval_positions.p`.

#### 1. Static Environment

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation.launch.py headless:=true

# Terminal 2 — TD3 LiDAR
ros2 run sddpg_navigation eval_td3 --model_name <run_name>

# Terminal 2 — Pure DDPG baseline
ros2 run sddpg_navigation eval_pure_ddpg --model_name <run_name>

# Terminal 2 — Legacy GRU DDPG (loads from evaluation/saved_model/)
ros2 run sddpg_navigation eval_ddpg

# Terminal 2 — Spiking DDPG (legacy)
ros2 run sddpg_navigation eval_sddpg
ros2 run sddpg_navigation eval_sddpg \
  --save_dir src/sddpg_navigation/sddpg_navigation/save_sddpg_weights/ \
  --model_name <run_name> --checkpoint <episode>
```

#### 2. Dynamic Obstacle Environment

The DVS evaluation requires `evaluation_dynamic.launch.py` because the `event_camera` node must be running.

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation_dynamic.launch.py headless:=true

# Terminal 2 — TD3 LiDAR
ros2 run sddpg_navigation eval_td3 --model_name <run_name>

# Terminal 2 — DVS
ros2 run sddpg_navigation eval_dvs --model_name <run_name> --checkpoint <step>
```

#### 3. Monitoring Dynamic Obstacle Distances

To verify that dynamic obstacle collision detection is active during a live run:

```bash
ros2 run sddpg_navigation test_dynamic_collision
```

Prints robot-to-obstacle distance for each obstacle at 5 Hz.
Distances below 0.35 m (training threshold) and 0.25 m (evaluation threshold) are flagged in the output.

## Package Structure

```
src/sddpg_navigation/
├── launch/
│   ├── training.launch.py                  — static world
│   ├── training_dynamic.launch.py          — dynamic world + event_camera + obstacle_mover
│   ├── training_dynamic_lidar.launch.py    — dynamic world + obstacle_mover
│   ├── evaluation.launch.py
│   ├── evaluation_dynamic.launch.py        — dynamic world + event_camera
│   └── bridge.yaml                         — ROS ↔ Gazebo topic bridge
├── models/turtlebot3_burger/               — robot SDF (LiDAR + camera sensor)
├── worlds/                                 — Gazebo world files
└── sddpg_navigation/
    ├── environment.py                      — GazeboEnvironment node: step(), reset()
    ├── utility.py                          — env generation, action decoding
    ├── event_camera.py                     — synthetic DVS event publisher
    ├── dynamic_obstacles.py                — obstacle mover node (20 Hz)
    ├── test_dynamic_collision.py           — live collision distance monitor
    ├── training/
    │   ├── train_pure_ddpg/                — DDPG baseline (FC actor)
    │   ├── train_td3/                      — TD3 LiDAR (GRU actor)
    │   ├── train_DVS_ddpg/                 — TD3 DVS (CNN+GRU actor)
    │   └── train_spiking_ddpg/             — SDDPG (legacy, Loihi)
    ├── evaluation/
    │   ├── eval_random_simulation/         — simulation evaluation scripts
    │   └── result_analyze/                 — result processing utilities
    └── random_positions/                   — pre-generated start/goal pairs (pickle)
```

## Legacy Code

The `original_code_docker/` directory contains the original Docker-based setup from the legacy ROS 1 / Gazebo 7 implementation. It is preserved for reference and is not required for running the current codebase.
