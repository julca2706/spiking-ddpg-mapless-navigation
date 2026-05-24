import rclpy
from rclpy.node import Node
import math
import time
import copy
import random
import torch
import numpy as np
from shapely.geometry import Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import sys
sys.path.insert(0, '/opt/ros/jazzy/opt/gz_msgs_vendor/lib/python')
from gz.msgs10.pose_v_pb2 import Pose_V
import gz.transport13 as gz_transport
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose, ControlWorld
from sddpg_navigation.evaluation.eval_random_simulation.utility import *
from sddpg_navigation.utility import (robot_2_goal_dis_dir,
                                      wheeled_network_2_robot_action_decoder,
                                      ddpg_state_2_spike_value_state,
                                      ddpg_state_rescale)


class RandEvalGpu(Node):
    """ Perform Random Evaluation on GPU """
    def __init__(self,
                 actor_net,
                 robot_init_pose_list,
                 goal_pos_list,
                 obstacle_poly_list,
                 ros_rate=10,
                 max_steps=4000,
                 min_spd=0.05,
                 max_spd=0.25,
                 is_spike=False,
                 is_scale=False,
                 is_poisson=False,
                 batch_window=50,
                 action_rand=0.05,
                 scan_half_num=9,
                 scan_min_dis=0.35,
                 goal_dis_min_dis=0.3,
                 goal_th=0.5,
                 obs_near_th=0.09,
                 use_cuda=True,
                 is_record=False):
        
        super().__init__('random_evaluation_gpu')
        """
        :param actor_net: Actor Network
        :param robot_init_pose_list: robot init pose list
        :param goal_pos_list: goal position list
        :param obstacle_poly_list: obstacle list
        :param ros_rate: ros rate
        :param max_steps: max step for single goal
        :param min_spd: min wheel speed
        :param max_spd: max wheel speed
        :param is_spike: is using SNN
        :param is_scale: is scale DDPG state input
        :param is_poisson: is use rand DDPG state input
        :param batch_window: batch window of SNN
        :param action_rand: random of action
        :param scan_half_num: half number of scan points
        :param scan_min_dis: min distance of scan
        :param goal_dis_min_dis: min distance of goal distance
        :param goal_th: distance for reach goal
        :param obs_near_th: distance for obstacle collision
        :param use_cuda: if true use cuda
        :param is_record: if true record running data
        """
        self.actor_net = actor_net
        self.robot_init_pose_list = robot_init_pose_list
        self.goal_pos_list = goal_pos_list
        self.obstacle_poly_list = obstacle_poly_list
        self.ros_rate = ros_rate
        self.max_steps = max_steps
        self.min_spd = min_spd
        self.max_spd = max_spd
        self.is_spike = is_spike
        self.is_scale = is_scale
        self.is_poisson = is_poisson
        self.batch_window = batch_window
        self.action_rand = action_rand
        self.scan_half_num = scan_half_num
        self.scan_min_dis = scan_min_dis
        self.goal_dis_min_dis = goal_dis_min_dis
        self.goal_th = goal_th
        self.obs_near_th = obs_near_th
        self.use_cuda = use_cuda
        self.is_record = is_record
        self.record_data = []

        # Put network to device
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.actor_net.to(self.device)

        # Robot State
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_pose = [0., 0., 0.]
        self.robot_spd = [0., 0.]
        self.robot_scan = np.zeros(2 * scan_half_num)
        self.robot_world_pose_init = False

        # Dynamic obstacle positions (published by dynamic_obstacles.py)
        self.dynamic_obstacle_positions = []

        # Subscriber
        self.create_subscription(Odometry, '/odom', self._robot_state_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._robot_scan_cb, 10)
        self.create_subscription(Float32MultiArray, '/dynamic_obstacle_positions',
                                 self._dynamic_obs_cb, 10)

        # Direct gz-transport subscriber for world-frame pose
        self._gz_node = gz_transport.Node()
        self._gz_node.subscribe(Pose_V, '/world/default/dynamic_pose/info', self._robot_world_pose_cb)

        # Publisher
        self.pub_action = self.create_publisher(Twist, 'cmd_vel', 10)

        # Service
        self.pause_gazebo = self.create_client(ControlWorld, '/world/default/control')
        self.unpause_gazebo = self.create_client(ControlWorld, '/world/default/control')
        self.set_model_target = self.create_client(SetEntityPose, '/world/default/set_pose')

        # Init Subscriber
        self.get_logger().info("Waiting for topics...")
        while not self.robot_state_init or not self.robot_scan_init or not self.robot_world_pose_init:
            rclpy.spin_once(self, timeout_sec=1.0)
        self.get_logger().info("Finish Subscriber Init...")

    def run_ros(self):
        """
        ROS ROS Node
        :return: run_data
        """
        run_num = len(self.robot_init_pose_list)
        run_data = {"final_state": np.zeros(run_num),
                    "time": np.zeros(run_num),
                    "path": []}
        goal_ita = 0
        single_goal_run_ita = 0
        failure_case = 0
        robot_path = []
        req = ControlWorld.Request()
        self._set_new_target(goal_ita)
        print("Test: ", goal_ita)
        print("Start Robot Pose: (%.3f, %.3f, %.3f) Goal: (%.3f, %.3f)" %
              (self.robot_init_pose_list[goal_ita][0], self.robot_init_pose_list[goal_ita][1],
               self.robot_init_pose_list[goal_ita][2],
               self.goal_pos_list[goal_ita][0], self.goal_pos_list[goal_ita][1]))
        print("  Actual pose after reset: (%.3f, %.3f, %.3f)" % tuple(self.robot_pose))
        print("  Scan min=%.3f max=%.3f mean=%.3f" % (
            float(np.min(self.robot_scan)), float(np.max(self.robot_scan)), float(np.mean(self.robot_scan))))
        goal_start_time = time.time()
        while rclpy.ok():
            tmp_robot_pose = copy.deepcopy(self.robot_pose)
            tmp_robot_spd = copy.deepcopy(self.robot_spd)
            tmp_robot_scan = copy.deepcopy(self.robot_scan)
            tmp_robot_scan[tmp_robot_scan == 0] = 0.001
            tmp_robot_scan = self.scan_min_dis / tmp_robot_scan
            tmp_robot_scan = np.clip(tmp_robot_scan, 0, 1)
            goal_dis, goal_dir = robot_2_goal_dis_dir(tmp_robot_pose, self.goal_pos_list[goal_ita])
            is_near_obs = self._near_obstacle(tmp_robot_pose)
            robot_path.append(tmp_robot_pose)
            '''
            Set new test goal
            '''
            if goal_dis < self.goal_th or is_near_obs or single_goal_run_ita == self.max_steps:
                goal_end_time = time.time()
                run_data['time'][goal_ita] = goal_end_time - goal_start_time
                if goal_dis < self.goal_th:
                    print("End: Success")
                    run_data['final_state'][goal_ita] = 1
                elif is_near_obs:
                    failure_case += 1
                    print("End: Obstacle Collision at step %d pose=(%.3f,%.3f) scan_min=%.3f" % (
                        single_goal_run_ita, tmp_robot_pose[0], tmp_robot_pose[1],
                        float(np.min(self.robot_scan))))
                    run_data['final_state'][goal_ita] = 2
                elif single_goal_run_ita == self.max_steps:
                    failure_case += 1
                    print("End: Out of steps")
                    run_data['final_state'][goal_ita] = 3
                print("Up to step failure number: ", failure_case)
                run_data['path'].append(robot_path)
                goal_ita += 1
                if goal_ita == run_num:
                    break
                single_goal_run_ita = 0
                robot_path = []
                self._set_new_target(goal_ita)
                print("Test: ", goal_ita)
                print("Start Robot Pose: (%.3f, %.3f, %.3f) Goal: (%.3f, %.3f)" %
                      (self.robot_init_pose_list[goal_ita][0], self.robot_init_pose_list[goal_ita][1],
                       self.robot_init_pose_list[goal_ita][2],
                       self.goal_pos_list[goal_ita][0], self.goal_pos_list[goal_ita][1]))
                goal_start_time = time.time()
                continue
            '''
            Perform Action
            '''
            tmp_goal_dis = goal_dis
            if tmp_goal_dis == 0:
                tmp_goal_dis = 1
            else:
                tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis
                if tmp_goal_dis > 1:
                    tmp_goal_dis = 1
            ddpg_state = [goal_dir, tmp_goal_dis, tmp_robot_spd[0], tmp_robot_spd[1]]
            ddpg_state.extend(tmp_robot_scan.tolist())
            action = self._network_2_robot_action(ddpg_state)

            while not self.unpause_gazebo.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Waiting for unpause service...")
            req.world_control.pause = False
            future = self.unpause_gazebo.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            move_cmd = Twist()
            move_cmd.linear.x = action[0]
            move_cmd.angular.z = action[1]
            self.pub_action.publish(move_cmd)
            deadline = time.time() + (1.0 / self.ros_rate)
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.01)

            while not self.pause_gazebo.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Waiting for pause service...")
            req.world_control.pause = True
            future = self.pause_gazebo.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.01)

            single_goal_run_ita += 1
        suc_num = np.sum(run_data["final_state"] == 1)
        obs_num = np.sum(run_data["final_state"] == 2)
        out_num = np.sum(run_data["final_state"] == 3)
        print("Success: ", suc_num, " Obstacle Collision: ", obs_num, " Over Steps: ", out_num)
        print("Success Rate: ", suc_num / run_num)
        return run_data

    def _network_2_robot_action(self, state):
        """
        Generate robot action based on network output
        :param state: ddpg state
        :return: [linear spd, angular spd]
        """
        with torch.no_grad():
            if self.is_spike:
                state = self._state_2_state_spikes(state)
                if self.is_record:
                    self.record_data.append(state)
                state = torch.Tensor(state).to(self.device)
                action = self.actor_net(state, 1).to('cpu')
            elif self.is_scale:
                state = self._state_2_scale_state(state)
                if self.is_record:
                    self.record_data.append(state)
                state = torch.Tensor(state).to(self.device)
                action = self.actor_net(state).to('cpu')
            else:
                state = np.array(state).reshape((1, -1))
                if self.is_record:
                    self.record_data.append(state)
                state = torch.Tensor(state).to(self.device)
                action = self.actor_net(state).to('cpu')
            action = action.numpy().squeeze()
        noise = np.random.randn(2) * self.action_rand
        action = noise + (1 - self.action_rand) * action
        action = np.clip(action, [0., 0.], [1., 1.])
        action = wheeled_network_2_robot_action_decoder(
            action, self.max_spd, self.min_spd
        )
        return action

    def _state_2_state_spikes(self, state):
        """
        Transform state to spikes of input neurons
        :param state: robot state
        :return: state_spikes
        """
        spike_state_num = self.scan_half_num * 2 + 6
        spike_state_value = ddpg_state_2_spike_value_state(state, spike_state_num)
        spike_state_value = np.array(spike_state_value)
        spike_state_value = spike_state_value.reshape((1, spike_state_num, 1))
        state_spikes = np.random.rand(1, spike_state_num, self.batch_window) < spike_state_value
        state_spikes = state_spikes.astype(float)
        return state_spikes

    def _state_2_scale_state(self, state):
        """
        Transform state to scale state with or without Poisson random
        :param state: robot state
        :return: scale_state
        """ 
        if self.is_poisson:
            scale_state_num = self.scan_half_num * 2 + 6
            state = ddpg_state_2_spike_value_state(state, scale_state_num)
            state = np.array(state)
            spike_state_value = state.reshape((1, scale_state_num, 1))
            state_spikes = np.random.rand(1, scale_state_num, self.batch_window) < spike_state_value
            poisson_state = np.sum(state_spikes, axis=2).reshape((1, -1))
            poisson_state = poisson_state / self.batch_window
            scale_state = poisson_state.astype(float)
        else:
            scale_state_num = self.scan_half_num * 2 + 4
            scale_state = ddpg_state_rescale(state, scale_state_num)
            scale_state = np.array(scale_state).reshape((1, scale_state_num))
            scale_state = scale_state.astype(float)
        return scale_state

    def _near_obstacle(self, pos):
        """
        Test if robot is near obstacle (static or dynamic).
        :param pos: robot position
        :return: done
        """
        done = False
        robot_point = Point(pos[0], pos[1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                done = True
                break
        
        if not done:
            for (ox, oy) in self.dynamic_obstacle_positions:
                if robot_point.distance(Point(ox, oy)) < 0.25:
                    done = True
                    break
        return done

    def _dynamic_obs_cb(self, msg):
        data = msg.data
        self.dynamic_obstacle_positions = [
            (data[i], data[i + 1]) for i in range(0, len(data), 2)
        ]

    def _set_new_target(self, ita):
        """
        Set new robot pose and goal position, mirroring training reset().
        Unpauses sim for teleport, waits for state to settle, then pauses.
        :param ita: goal ita
        """
        goal_position = self.goal_pos_list[ita]
        req = ControlWorld.Request()

        while not self.unpause_gazebo.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for unpause service...")
        req.world_control.pause = False
        future = self.unpause_gazebo.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        while not self.set_model_target.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for set_pose service...")

        request = SetEntityPose.Request()
        request.entity.name = 'target'
        request.entity.type = 2
        request.pose.position.x = float(goal_position[0])
        request.pose.position.y = float(goal_position[1])
        request.pose.position.z = 0.0
        future = self.set_model_target.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        self.pub_action.publish(Twist())
        robot_init_pose = self.robot_init_pose_list[ita]
        robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[2])
        request = SetEntityPose.Request()
        request.entity.name = 'turtlebot3_burger'
        request.entity.type = 2
        request.pose.position.x = float(robot_init_pose[0])
        request.pose.position.y = float(robot_init_pose[1])
        request.pose.orientation.x = float(robot_init_quat[1])
        request.pose.orientation.y = float(robot_init_quat[2])
        request.pose.orientation.z = float(robot_init_quat[3])
        request.pose.orientation.w = float(robot_init_quat[0])
        future = self.set_model_target.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.2)

        while not self.pause_gazebo.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for pause service...")
        req.world_control.pause = True
        future = self.pause_gazebo.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        rclpy.spin_once(self, timeout_sec=0.1)

    def _euler_2_quat(self, yaw=0, pitch=0, roll=0):
        """
        Transform euler angule to quaternion
        :param yaw: z
        :param pitch: y
        :param roll: x
        :return: quaternion
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [w, x, y, z]

    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """
        if self.robot_state_init is False:
            self.robot_state_init = True
        linear_spd = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.robot_spd = [linear_spd, msg.twist.twist.angular.z]

    def _robot_scan_cb(self, msg):
        """
        Callback function for robot laser scan
        :param msg: message
        """
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
        scan_dir_num = self.scan_half_num * 4
        chunk = len(ranges) // scan_dir_num
        full_scan = np.array([
            np.median(ranges[i * chunk:(i + 1) * chunk])
            for i in range(scan_dir_num)
        ])
        scan_center = scan_dir_num // 2  # = 18
        tmp_ita = 0
        for num in range(self.scan_half_num):
            ita = scan_center + self.scan_half_num - num - 1  # 26, 25, ..., 18
            self.robot_scan[tmp_ita] = full_scan[ita]
            tmp_ita += 1
        for num in range(self.scan_half_num):
            ita = scan_center - num - 1  # 17, 16, ..., 9
            self.robot_scan[tmp_ita] = full_scan[ita]
            tmp_ita += 1
    
    def _robot_world_pose_cb(self, msg):
        for pose in msg.pose:
            if pose.name == 'turtlebot3_burger':
                if not self.robot_world_pose_init:
                    self.robot_world_pose_init = True
                p = pose.position
                r = pose.orientation
                siny_cosp = 2. * (r.x * r.y + r.z * r.w)
                cosy_cosp = 1. - 2. * (r.y ** 2 + r.z ** 2)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                self.robot_pose = [p.x, p.y, yaw]
                break

