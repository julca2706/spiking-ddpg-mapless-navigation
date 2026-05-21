from setuptools import find_packages, setup

package_name = 'sddpg_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/training.launch.py',
            'launch/evaluation.launch.py',
            'launch/training_dynamic.launch.py',
            'launch/evaluation_dynamic.launch.py',
            'launch/bridge.yaml']),
        ('share/' + package_name + '/worlds', [
            'worlds/training_worlds.world',
            'worlds/evaluation_world.world',
            'worlds/training_dynamic_world.world',
            'worlds/evaluation_dynamic_world.world']),
        ('share/' + package_name + '/models/turtlebot3_burger', [
            'models/turtlebot3_burger/model.sdf',
            'models/turtlebot3_burger/model.config']),
        ('share/' + package_name + '/models/textures', [
            'models/textures/model.config',
            'models/textures/vertical_stripes.png'])],
    package_data={
        'sddpg_navigation': ['random_positions/*.p'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Julia Caluch',
    maintainer_email='j.cauch@student.maastrichtuniversity.nl',
    description='Spiking DDPG mapless navigation - ROS 2 port',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'environment = sddpg_navigation.environment:main',
            'train_sddpg = sddpg_navigation.training.train_spiking_ddpg.train_sddpg:main',
            'train_ddpg = sddpg_navigation.training.train_ddpg.train_ddpg:main',
            'train_td3 = sddpg_navigation.training.train_td3.train_td3:main',
            'eval_sddpg = sddpg_navigation.evaluation.eval_random_simulation.run_sddpg_eval:main',
            'eval_ddpg = sddpg_navigation.evaluation.eval_random_simulation.run_ddpg_eval:main',
            'obstacle_mover_node = sddpg_navigation.dynamic_obstacles:main',
            'event_camera = sddpg_navigation.event_camera:main',
            'train_dvs_ddpg = sddpg_navigation.training.train_DVS_ddpg.train_DVS_ddpg:main',
            'train_pure_ddpg = sddpg_navigation.training.train_pure_ddpg.train_ddpg:main',
            'eval_pure_ddpg = sddpg_navigation.evaluation.eval_random_simulation.run_pure_ddpg_eval:main',
            'eval_gru_ddpg = sddpg_navigation.evaluation.eval_random_simulation.run_gru_ddpg_eval:main',
        ],
    },
)
