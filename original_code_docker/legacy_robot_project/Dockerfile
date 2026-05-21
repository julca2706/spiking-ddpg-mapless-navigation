FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    wget curl git vim build-essential software-properties-common \
    udev kmod \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list' && \
    (apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 || \
     wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | apt-key add -)

RUN apt-get update && apt-get install -y \
    ros-kinetic-desktop-full \
    python-rosdep python-rosinstall python-rosinstall-generator python-wstool \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri libgl1-mesa-glx mesa-utils libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-kinetic-kobuki-core \
    ros-kinetic-kobuki-description \
    ros-kinetic-kobuki-gazebo-plugins \
    ros-kinetic-kobuki-gazebo \
    ros-kinetic-kobuki-bumper2pc \
    ros-kinetic-urg-node \
    ros-kinetic-tf ros-kinetic-tf-conversions \
    ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    (apt-get install -y \
        ros-kinetic-turtlebot \
        ros-kinetic-turtlebot-bringup \
        ros-kinetic-turtlebot-description \
        ros-kinetic-turtlebot-teleop \
        ros-kinetic-turtlebot-rviz-launchers \
    || true) && \
    rm -rf /var/lib/apt/lists/*

RUN dpkg --configure -a 2>/dev/null || true && \
    dpkg --remove --force-remove-reinstreq \
        ros-kinetic-librealsense \
        ros-kinetic-realsense-camera \
        ros-kinetic-turtlebot-bringup \
        ros-kinetic-turtlebot-teleop \
        ros-kinetic-turtlebot \
        ros-kinetic-turtlebot-rviz-launchers \
    2>/dev/null || true && \
    apt-get update && apt-get install -f -y && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

RUN apt-get update && apt-get install -y \
    python3.5 python3.5-dev python3-pip python-pip virtualenv \
    && rm -rf /var/lib/apt/lists/*

RUN python2 -m pip install --upgrade "pip==20.3.4" "setuptools==44.1.1" "wheel==0.37.1"

RUN python3 -m pip install --upgrade "pip==20.3.4" "setuptools==50.3.2" "wheel==0.37.1"

RUN python2 -m pip install "numpy==1.16.6" "Shapely==1.7.0"

RUN mkdir -p /root/ros/catkin_ws/src
WORKDIR /root/ros/catkin_ws

COPY project_files/ros/catkin_ws/src/simple_laserscan    /root/ros/catkin_ws/src/simple_laserscan
COPY project_files/ros/catkin_ws/src/turtlebot_description /root/ros/catkin_ws/src/turtlebot_description
COPY project_files/ros/catkin_ws/src/turtlebot_lidar      /root/ros/catkin_ws/src/turtlebot_lidar

RUN mkdir -p /root/ros/catkin_ws/src/turtlebot_bringup/param \
             /root/ros/catkin_ws/src/turtlebot_bringup/launch/includes/kobuki

RUN printf '<package>\n  <name>turtlebot_bringup</name>\n  <version>2.3.3</version>\n  <description>Stub turtlebot_bringup for simulation</description>\n  <maintainer email="stub@stub.com">stub</maintainer>\n  <license>BSD</license>\n  <buildtool_depend>catkin</buildtool_depend>\n</package>\n' \
    > /root/ros/catkin_ws/src/turtlebot_bringup/package.xml && \
    printf 'cmake_minimum_required(VERSION 2.8.3)\nproject(turtlebot_bringup)\nfind_package(catkin REQUIRED)\ncatkin_package()\n' \
    > /root/ros/catkin_ws/src/turtlebot_bringup/CMakeLists.txt

RUN printf 'lock_timeout: 0.0\nsubscribers:\n\
  - name: "Safety Controller"\n    topic: "/cmd_vel_mux/input/safety_controller"\n    timeout: 0.2\n    priority: 10\n\
  - name: "Teleop"\n    topic: "/cmd_vel_mux/input/teleop"\n    timeout: 0.2\n    priority: 8\n\
  - name: "Navigation"\n    topic: "/cmd_vel_mux/input/navi"\n    timeout: 0.2\n    priority: 6\n\
  - name: "Joystick"\n    topic: "/cmd_vel_mux/input/joystick"\n    timeout: 0.2\n    priority: 4\n' \
    > /root/ros/catkin_ws/src/turtlebot_bringup/param/mux.yaml

RUN printf '<launch>\n</launch>\n' \
    > /root/ros/catkin_ws/src/turtlebot_bringup/launch/includes/kobuki/bumper2pc.launch.xml

RUN mkdir -p /usr/share/gazebo-7/models/sun \
             /usr/share/gazebo-7/models/ground_plane

RUN printf '<?xml version="1.0"?>\n<model>\n  <name>sun</name>\n  <version>1.0</version>\n  <sdf version="1.5">model.sdf</sdf>\n  <description>Sun directional light.</description>\n</model>\n' \
    > /usr/share/gazebo-7/models/sun/model.config && \
    printf '<?xml version="1.0"?>\n<sdf version="1.5">\n  <model name="sun">\n    <static>true</static>\n    <link name="link">\n      <visual name="visual">\n        <geometry><sphere><radius>0.1</radius></sphere></geometry>\n      </visual>\n    </link>\n    <light name="sun" type="directional">\n      <cast_shadows>1</cast_shadows>\n      <pose>0 0 10 0 0 0</pose>\n      <diffuse>0.8 0.8 0.8 1</diffuse>\n      <specular>0.2 0.2 0.2 1</specular>\n      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>\n      <direction>-0.5 0.1 -0.9</direction>\n    </light>\n  </model>\n</sdf>\n' \
    > /usr/share/gazebo-7/models/sun/model.sdf

RUN printf '<?xml version="1.0"?>\n<model>\n  <name>Ground Plane</name>\n  <version>1.0</version>\n  <sdf version="1.4">model.sdf</sdf>\n  <description>A flat ground plane.</description>\n</model>\n' \
    > /usr/share/gazebo-7/models/ground_plane/model.config && \
    printf '<?xml version="1.0"?>\n<sdf version="1.4">\n  <model name="ground_plane">\n    <static>true</static>\n    <link name="link">\n      <collision name="collision">\n        <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>\n        <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>\n      </collision>\n      <visual name="visual">\n        <cast_shadows>false</cast_shadows>\n        <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>\n        <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Grey</name></script></material>\n      </visual>\n    </link>\n  </model>\n</sdf>\n' \
    > /usr/share/gazebo-7/models/ground_plane/model.sdf

RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash && catkin_make"

RUN virtualenv -p python3.5 /root/py3_env

RUN /bin/bash -c "source /root/py3_env/bin/activate && \
    pip install --upgrade pip==20.3.4 setuptools==50.3.2 wheel==0.37.1"

RUN /bin/bash -c "source /root/py3_env/bin/activate && \
    pip install torch==1.2.0 torchvision==0.4.0 \
        -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install tensorboard==2.1.0 && \
    pip install numpy==1.18.5 Shapely==1.7.0 Pillow==7.2.0 && \
    pip install rospkg catkin-pkg future netifaces"

ENV TURTLEBOT_3D_SENSOR=hokuyo
ENV ROS_MASTER_URI=http://localhost:11311
ENV ROS_HOSTNAME=localhost
ENV GAZEBO_MODEL_PATH=/usr/share/gazebo-7/models

RUN echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "source /usr/share/gazebo-7/setup.sh" >> /root/.bashrc && \
    echo "export TURTLEBOT_3D_SENSOR=hokuyo" >> /root/.bashrc && \
    echo "alias activate_py3='source /root/py3_env/bin/activate'" >> /root/.bashrc && \
    echo "alias goto_project='cd /root/project'" >> /root/.bashrc && \
    echo "# Make ROS packages importable from the Python 3 virtualenv" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/kinetic/lib/python2.7/dist-packages:/root/ros/catkin_ws/devel/lib/python2.7/dist-packages:\$PYTHONPATH" >> /root/.bashrc

RUN printf '#!/bin/bash\n\
echo "========================================"\n\
echo "ROS Kinetic Environment Ready"\n\
echo "========================================"\n\
echo "ROS (Python 2.7) active by default"\n\
echo "Project: /root/project"\n\
echo "activate_py3  - switch to Python 3.5"\n\
echo "goto_project  - cd to project"\n\
echo "========================================"\n\
exec "$@"\n' > /root/entrypoint.sh && chmod +x /root/entrypoint.sh

WORKDIR /root/ros/catkin_ws
ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["/bin/bash"]
