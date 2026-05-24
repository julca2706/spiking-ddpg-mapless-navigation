import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Launches: Gazebo (dynamic world, env1 + env4 with moving obstacles) + ROS-Gazebo bridge
#           + event_camera node (DVS processing, required for DVS actor training)
#           + obstacle_mover node (moves obstacles, publishes /dynamic_obstacle_positions).
# For all specifications of existing launches check README.md.

def generate_launch_description():
    pkg_share = get_package_share_directory('sddpg_navigation')
    world_file = os.path.join(pkg_share, 'worlds', 'training_dynamic_world.world')
    bridge_config = os.path.join(pkg_share, 'launch', 'bridge.yaml')
    models_dir = os.path.join(pkg_share, 'models')
    headless = LaunchConfiguration('headless', default='false')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': PythonExpression([
                '"-r -s ' + world_file + '" if "', headless, '" == "true" else "-r ' + world_file + '"'
            ])
        }.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='bridge_node',
        name='ros_gz_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
    )

    event_camera = Node(
        package='sddpg_navigation',
        executable='event_camera',
        name='event_camera_node',
        output='screen',
    )

    obstacle_mover = Node(
        package='sddpg_navigation',
        executable='obstacle_mover_node',
        name='obstacle_mover_node',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', models_dir),
        gz_sim,
        bridge,
        event_camera,
        obstacle_mover,
    ])
