import rclpy
from rclpy.node import Node
import math
import copy
import random
import time
import numpy as np
from shapely.geometry import Point
import sys
sys.path.insert(0, '/opt/ros/jazzy/opt/gz_msgs_vendor/lib/python')
from gz.msgs10.pose_v_pb2 import Pose_V
import gz.transport13 as gz_transport
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from ros_gz_interfaces.srv import SetEntityPose, ControlWorld


class GazeboEnvironment(Node):
    """
    Class for Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,
                 laser_scan_half_num=9,
                 laser_scan_min_dis=0.35,
                 laser_scan_scale=1.0,
                 scan_dir_num=36,
                 goal_dis_min_dis=0.5,
                 goal_dis_scale=1.0,
                 obs_near_th=0.35,
                 goal_near_th=0.5,
                 goal_reward=10,
                 obs_reward=-5,
                 goal_dis_amp=5,
                 step_time=0.1):
        
        super().__init__('gazebo_environment')
        
        """

        :param laser_scan_half_num: half number of scan points
        :param laser_scan_min_dis: Min laser scan distance
        :param laser_scan_scale: laser scan scale
        :param scan_dir_num: number of directions in laser scan
        :param goal_dis_min_dis: minimal distance of goal distance
        :param goal_dis_scale: goal distance scale
        :param obs_near_th: Threshold for near an obstacle
        :param goal_near_th: Threshold for near an goal
        :param goal_reward: reward for reaching goal
        :param obs_reward: reward for reaching obstacle
        :param goal_dis_amp: amplifier for goal distance change
        :param step_time: time for a single step (DEFAULT: 0.1 seconds)
        """
        self.goal_pos_list = None
        self.obstacle_poly_list = None
        self.robot_init_pose_list = None
        self.laser_scan_half_num = laser_scan_half_num
        self.laser_scan_min_dis = laser_scan_min_dis
        self.laser_scan_scale = laser_scan_scale
        self.scan_dir_num = scan_dir_num
        self.goal_dis_min_dis = goal_dis_min_dis
        self.goal_dis_scale = goal_dis_scale
        self.obs_near_th = obs_near_th
        self.goal_near_th = goal_near_th
        self.goal_reward = goal_reward
        self.obs_reward = obs_reward
        self.goal_dis_amp = goal_dis_amp
        self.step_time = step_time

        # Robot State
        self.robot_pose = [0., 0., 0.]
        self.robot_speed = [0., 0.]
        self.robot_scan = np.zeros(self.scan_dir_num)
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_world_pose_init = False

        # Goal Position
        self.goal_position = [0., 0.]
        self.goal_dis_dir_pre = [0., 0.]  # Last step goal distance and direction
        self.goal_dis_dir_cur = [0., 0.]  # Current step goal distance and direction

        # Subscriber
        self.create_subscription(Odometry, '/odom', self._robot_state_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._robot_scan_cb, 10)

        # Direct gz-transport subscriber for world-frame pose
        self._gz_node = gz_transport.Node()
        self._gz_node.subscribe(Pose_V, '/world/default/dynamic_pose/info', self._robot_world_pose_cb)

        # Publisher
        self.pub_action = self.create_publisher( Twist, 'cmd_vel', 10)

        # Service
        self.pause_gazebo = self.create_client(ControlWorld, '/world/default/control')
        self.unpause_gazebo = self.create_client(ControlWorld, '/world/default/control')
        self.set_model_target = self.create_client(SetEntityPose, '/world/default/set_pose')

        # Init Subscriber
        self.get_logger().info("Waiting for topics...")
        while not self.robot_state_init or not self.robot_scan_init or not self.robot_world_pose_init:
            rclpy.spin_once(self, timeout_sec=1.0)
        self.get_logger().info("Finish Subscriber Init...")

    def step(self, action):
        """
        Step Function for the Environment

        Take a action for the robot and return the updated state
        :param action: action taken
        :return: state, reward, done
        """
        assert self.goal_pos_list is not None
        assert self.obstacle_poly_list is not None
        req = ControlWorld.Request()

        while not self.unpause_gazebo.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for unpause service...")
        req.world_control.pause = False
        future = self.unpause_gazebo.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        '''
        First give action to robot and let robot execute and get next state
        '''
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]
        self.pub_action.publish(move_cmd)
        deadline = time.time() + self.step_time
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
        next_rob_state = self._get_next_robot_state()
        while not self.pause_gazebo.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for pause service...")
        req.world_control.pause = True
        future = self.pause_gazebo.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        '''
        Then stop the simulation
        1. Transform Robot State to DDPG State
        2. Compute Reward of the action
        3. Compute if the episode is ended
        '''
        goal_dis, goal_dir = self._compute_dis_dir_2_goal(next_rob_state[0])
        self.goal_dis_dir_cur = [goal_dis, goal_dir]
        state = self._robot_state_2_ddpg_state(next_rob_state)
        reward, done = self._compute_reward(next_rob_state)
        self.goal_dis_dir_pre = [self.goal_dis_dir_cur[0], self.goal_dis_dir_cur[1]]
        return state, reward, done

    def reset(self, ita):
        """
        Reset Function to reset simulation at start of each episode

        Return the initial state after reset
        :param ita: number of route to reset to
        :return: state
        """
        assert self.goal_pos_list is not None
        assert self.obstacle_poly_list is not None
        assert self.robot_init_pose_list is not None
        assert ita < len(self.goal_pos_list)

        req = ControlWorld.Request()

        while not self.unpause_gazebo.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for unpause service...")
        req.world_control.pause = False
        future = self.unpause_gazebo.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        '''
        First choose new goal position and set target model to goal
        '''
        self.goal_position = self.goal_pos_list[ita]
        request = SetEntityPose.Request()
        request.entity.name = 'target' 
        request.entity.type = 2
        request.pose.position.x = float(self.goal_position[0])
        request.pose.position.y = float(self.goal_position[1])
        request.pose.position.z = 0.0
        future = self.set_model_target.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        '''
        Then reset robot state and get initial state
        '''
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

        deadline = time.time() + 0.5
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
        '''
        Transfer the initial robot state to the state for the DDPG Agent
        '''
        rob_state = self._get_next_robot_state()
        while not self.pause_gazebo.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for pause service...")
        req.world_control.pause = True
        future = self.pause_gazebo.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        goal_dis, goal_dir = self._compute_dis_dir_2_goal(rob_state[0])
        self.goal_dis_dir_pre = [goal_dis, goal_dir]
        self.goal_dis_dir_cur = [goal_dis, goal_dir]
        state = self._robot_state_2_ddpg_state(rob_state)
        return state

    def set_new_environment(self, init_pose_list, goal_list, obstacle_list):
        """
        Set New Environment for training
        :param init_pose_list: init pose list of robot
        :param goal_list: goal position list
        :param obstacle_list: obstacle list
        """
        self.robot_init_pose_list = init_pose_list
        self.goal_pos_list = goal_list
        self.obstacle_poly_list = obstacle_list

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

    def _compute_dis_dir_2_goal(self, pose):
        """
        Compute the difference of distance and direction to goal position
        :param pose: pose of the robot
        :return: distance, direction
        """
        delta_x = self.goal_position[0] - pose[0]
        delta_y = self.goal_position[1] - pose[1]
        distance = math.sqrt(delta_x**2 + delta_y**2)
        ego_direction = math.atan2(delta_y, delta_x)
        robot_direction = pose[2]
        while robot_direction < 0:
            robot_direction += 2 * math.pi
        while robot_direction > 2 * math.pi:
            robot_direction -= 2 * math.pi
        while ego_direction < 0:
            ego_direction += 2 * math.pi
        while ego_direction > 2 * math.pi:
            ego_direction -= 2 * math.pi
        pos_dir = abs(ego_direction - robot_direction)
        neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
        if pos_dir <= neg_dir:
            direction = math.copysign(pos_dir, ego_direction - robot_direction)
        else:
            direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
        return distance, direction

    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time

        State will be: [robot_pose, robot_spd, scan]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_spd = copy.deepcopy(self.robot_speed)
        tmp_robot_scan = copy.deepcopy(self.robot_scan)
        state = [tmp_robot_pose, tmp_robot_spd, tmp_robot_scan]
        return state

    def _robot_state_2_ddpg_state(self, state):
        """
        Transform robot state to DDPG state
        Robot State: [robot_pose, robot_spd, scan]
        DDPG state: [Distance to goal, Direction to goal, Linear Spd, Angular Spd, Scan]
        :param state: robot state
        :return: ddpg_state
        """
        tmp_goal_dis = self.goal_dis_dir_cur[0]
        if tmp_goal_dis == 0:
            tmp_goal_dis = self.goal_dis_scale
        else:
            tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis
            if tmp_goal_dis > 1:
                tmp_goal_dis = 1
            tmp_goal_dis = tmp_goal_dis * self.goal_dis_scale
        ddpg_state = [self.goal_dis_dir_cur[1], tmp_goal_dis, state[1][0], state[1][1]]
        '''
        Transform distance in laser scan to [0, scale]
        '''
        tmp_laser_scan = self.laser_scan_scale * (self.laser_scan_min_dis / state[2])
        tmp_laser_scan = np.clip(tmp_laser_scan, 0, self.laser_scan_scale)
        # chunk 0 = angle_min ≈ -π (behind), chunk scan_dir_num//2 = forward.
        # Select front hemisphere centred at scan_center, left-to-right order.
        scan_center = self.scan_dir_num // 2
        for num in range(self.laser_scan_half_num):
            ita = scan_center + self.laser_scan_half_num - num - 1
            ddpg_state.append(tmp_laser_scan[ita])
        for num in range(self.laser_scan_half_num):
            ita = scan_center - num - 1
            ddpg_state.append(tmp_laser_scan[ita])
        return ddpg_state

    def _compute_reward(self, state):
        """
        Compute Reward of the action base on current DDPG state and last step goal distance and direction

        Reward:
            1. R_Arrive If Distance to Goal is smaller than D_goal
            2. R_Collision If Distance to Obstacle is smaller than D_obs
            3. a * (Last step distance to goal - current step distance to goal)

        If robot near obstacle then done
        :param state: DDPG state
        :return: reward, done
        """
        done = False
        '''
        First compute distance to all obstacles
        '''
        near_obstacle = False
        robot_point = Point(state[0][0], state[0][1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                near_obstacle = True
                break
        '''
        Assign Rewards
        '''
        if self.goal_dis_dir_cur[0] < self.goal_near_th:
            reward = self.goal_reward
            done = True
        elif near_obstacle:
            reward = self.obs_reward
            done = True
        else:
            reward = self.goal_dis_amp * (self.goal_dis_dir_pre[0] - self.goal_dis_dir_cur[0])
        return reward, done

    def _robot_state_cb(self, msg):
        if self.robot_state_init is False:
            self.robot_state_init = True
        linear_spd = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.robot_speed = [linear_spd, msg.twist.twist.angular.z]

    def _robot_scan_cb(self, msg):
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
        chunk = len(ranges) // self.scan_dir_num
        self.robot_scan = np.array([
            np.median(ranges[i * chunk:(i + 1) * chunk])
            for i in range(self.scan_dir_num)
        ])

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

    
def main(args=None):
    rclpy.init(args=args)
    env = GazeboEnvironment()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

