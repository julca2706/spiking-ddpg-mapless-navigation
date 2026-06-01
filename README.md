# Extending SDDPG Towards Dynamic Mapless Navigation with Temporal Memory and Event-Based Vision

This project was prepared in partial fulfillment of the requirements for the Degree of Bachelor of Computer Science, Maastricht University. It is an extension of the original SDDPG framework by Tang et al. ([IROS 2020](https://ieeexplore.ieee.org/abstract/document/9340948)).

The baseline is built on top of the original SDDPG codebase, modernized to ROS 2 Jazzy and Gazebo Harmonic, with additional agent variants, launchable worlds, and helper nodes. One significant difference is the use of TurtleBot3 Burger with an RGB camera andLiDAR sensor. The robot configuration was sourced from the official TurtleBot3 simulation repository ([link](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)).

## Agent Variants

- **DDPG** — Deep Deterministic Policy Gradient; original package, ported and evaluated.
- **SDDPG** — Spiking DDPG with LIF-neuron actor; original package, ported and evaluated.
- **TD3+GRU** — Twin Delayed DDPG with GRU actor and sequential replay buffer; fully trained and evaluated.
- **DVS (proposed)** — TD3 with CNN+GRU actor using synthetic event frames; LiDAR critic retained. Infrastructure implemented, training not stabilized.

## World Variants

- **Static Training** — Original training arenas with curriculum learning.
- **Static Evaluation** — Original evaluation arena.
- **Dynamic Training** — Extension of static training; dynamic obstacles move in random directions, changing every 5 seconds. Obstacles feature black-and-white striped textures to provide contrast for the event camera.
- **Dynamic Evaluation** — Extension of static evaluation; dynamic obstacles move randomly within designated sub-areas. Most obstacles also feature striped textures.

## Helper Nodes

- **Event Camera Node** — Generates and publishes synthetic event frames from RGB camera input using frame differencing. Threshold is adjustable.
- **Obstacle Mover Node** — Moves dynamic obstacles in random directions within the training or evaluation arena, changing direction at fixed intervals. Passes through static obstacles; triggers collision with the robot.

## Software Installation

#### 1. System Requirements

All requirements are in a sepatare file requirements.txt.

* Ubuntu 24.04 LTS
* Python 3.12
* ROS 2 Jazzy + Gazebo Harmonic
* PyTorch >= 2.0

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

Add both `source` lines to `~/.bashrc` to avoid repeating them in every terminal. If not, every newly open terminal needs:
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
```

After any code change, rebuild with `colcon build --symlink-install`.

## Training

Each training run requires two terminals. Start Gazebo first, 
then the agent. The `headless:=true` flag is optional — omit it 
to launch with the Gazebo GUI.

---

#### 1. DDPG

```bash
# Terminal 1
ros2 launch sddpg_navigation training.launch.py

# Terminal 2
ros2 run sddpg_navigation train_pure_ddpg
```

Weights saved to `save_pure_ddpg_weights/`.

---

#### 2. SDDPG

```bash
# Terminal 1
ros2 launch sddpg_navigation training.launch.py

# Terminal 2
ros2 run sddpg_navigation train_sddpg
```

Weights saved to `save_sddpg_weights/`.

---

#### 3. TD3+GRU — static

```bash
# Terminal 1
ros2 launch sddpg_navigation training.launch.py

# Terminal 2
ros2 run sddpg_navigation train_td3

# Optional: start from a specific environment (0–3)
ros2 run sddpg_navigation train_td3 --start_env 1
```

Weights saved to `save_td3_weights/`.

---

#### 4. TD3+GRU — dynamic obstacles

```bash
# Terminal 1
ros2 launch sddpg_navigation training_dynamic_lidar.launch.py

# Terminal 2
ros2 run sddpg_navigation train_td3
```
Weights saved to `save_td3_weights/`.

---

#### 5. DVS Event Camera

The DVS launch file also starts the `event_camera` node and 
the obstacle mover.

```bash
# Terminal 1
ros2 launch sddpg_navigation training_dynamic.launch.py

# Terminal 2
ros2 run sddpg_navigation train_dvs_ddpg

# Optional: start from a specific environment (0–3)
ros2 run sddpg_navigation train_dvs_ddpg --start_env 1
```

Weights saved to `save_dvs_weights/` as `{run_name}_dvs_actor_s{step}.pt`.

To inspect event camera output (optional, separate terminal):

```bash
ros2 run rqt_image_view rqt_image_view
```

## Evaluation

Evaluation runs 200 fixed start/goal pairs from 
`evaluation/eval_random_simulation/eval_positions.p`.

The `headless:=true` flag is optional.

---

#### 1. DDPG

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation.launch.py

# Terminal 2
ros2 run sddpg_navigation eval_ddpg
```

---

#### 2. SDDPG

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation.launch.py

# Terminal 2
ros2 run sddpg_navigation eval_sddpg
# Or 
ros2 run sddpg_navigation eval_sddpg \
  --save_dir src/sddpg_navigation/sddpg_navigation/save_sddpg_weights/ \
  --model_name <run_name> --checkpoint <episode>
```

---

#### 3. TD3+GRU — static

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation.launch.py

# Terminal 2
ros2 run sddpg_navigation eval_td3 --model_name <run_name>
```

---

#### 4. TD3+GRU — dynamic

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation_dynamic.launch.py

# Terminal 2
ros2 run sddpg_navigation eval_td3 --model_name <run_name>
```

---

#### 5. DVS Event Camera — dynamic

The DVS evaluation requires `evaluation_dynamic.launch.py` 
because the `event_camera` node must be running.

```bash
# Terminal 1
ros2 launch sddpg_navigation evaluation_dynamic.launch.py

# Terminal 2
ros2 run sddpg_navigation eval_dvs --model_name <run_name>
```

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
