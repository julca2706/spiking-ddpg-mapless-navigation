import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import random, math


class ObstacleMoverNode(Node):
    """
    Moves dynamic obstacles in Gazebo at 20 Hz via the /world/default/set_pose service.

    Each obstacle moves at constant speed (0.2 m/s) in a random direction, 
    does not leav the area of training/evaluation - bounces off the borders,
    and randomly changes direction every 5 seconds.

    Current positions are published to /dynamic_obstacle_positions (Float32MultiArray,
    interleaved x/y pairs) for collision detection in the training and evaluation environments.

    Two obstacle sets are available, selected via the 'mode' ROS parameter:
      - 'training'   (default): 7 obstacles in env1 (4) and env4 (3)
      - 'evaluation': 6 obstacles spread across the evaluation arena

    A 5-second startup delay is applied to allow Gazebo to finish loading models
    before obstacle movement begins. Obstacles can be reset to their initial positions
    via the reset_obstacles service (std_srvs/Empty).
    """
    def __init__(self):
        super().__init__('obstacle_mover_node')

        self.cli = self.create_client(
            SetEntityPose, '/world/default/set_pose')
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /world/default/set_entity_pose...')

        self.declare_parameter('mode', 'training')
        mode = self.get_parameter('mode').get_parameter_value().string_value

        self.dt = 0.05          # 20 Hz
        self.speed = 0.2        # m/s
        self.redir_interval = 5.0

        training_obstacles = [
            # ENV1 (bottom-left)
            {'name': 'env1_dyn1', 'x0': -4.0,  'y0':  -5.0,
             'xmin': -13.5, 'xmax': -2.5, 'ymin': -13.5, 'ymax': -2.5},
            {'name': 'env1_dyn2', 'x0': -12.0, 'y0':  -5.0,
             'xmin': -13.5, 'xmax': -2.5, 'ymin': -13.5, 'ymax': -2.5},
            {'name': 'env1_dyn3', 'x0': -8.0,  'y0': -12.0,
             'xmin': -13.5, 'xmax': -2.5, 'ymin': -13.5, 'ymax': -2.5},
            {'name': 'env1_dyn4', 'x0': -4.0,  'y0': -12.0,
             'xmin': -13.5, 'xmax': -2.5, 'ymin': -13.5, 'ymax': -2.5},
            # ENV4 (top-left)
            {'name': 'env4_dyn1', 'x0': -3.0,  'y0':   8.0,
             'xmin': -13.5, 'xmax': -1.5, 'ymin':  2.5, 'ymax': 13.5},
            {'name': 'env4_dyn2', 'x0': -13.0, 'y0':   8.0,
             'xmin': -13.5, 'xmax': -1.5, 'ymin':  2.5, 'ymax': 13.5},
            {'name': 'env4_dyn3', 'x0': -8.0,  'y0':  13.0,
             'xmin': -13.5, 'xmax': -1.5, 'ymin':  2.5, 'ymax': 13.5},
        ]

        evaluation_obstacles = [
            {'name': 'eval_dyn1', 'x0': -7.0,  'y0':  7.0,
             'xmin': -9.0, 'xmax': -3.0, 'ymin':  3.0, 'ymax':  9.0},
            {'name': 'eval_dyn2', 'x0':  7.0,  'y0':  7.0,
             'xmin':  3.0, 'xmax':  9.0, 'ymin':  3.0, 'ymax':  9.0},
            {'name': 'eval_dyn3', 'x0': -3.0,  'y0':  0.0,
             'xmin': -6.0, 'xmax':  0.0, 'ymin': -3.0, 'ymax':  3.0},
            {'name': 'eval_dyn4', 'x0':  3.0,  'y0':  0.0,
             'xmin':  0.0, 'xmax':  6.0, 'ymin': -3.0, 'ymax':  3.0},
            {'name': 'eval_dyn5', 'x0': -7.0,  'y0': -7.0,
             'xmin': -9.0, 'xmax': -3.0, 'ymin': -9.0, 'ymax': -3.0},
            {'name': 'eval_dyn6', 'x0':  7.0,  'y0': -7.0,
             'xmin':  3.0, 'xmax':  9.0, 'ymin': -9.0, 'ymax': -3.0},
        ]

        self.obstacles = evaluation_obstacles if mode == 'evaluation' else training_obstacles
        
        for obs in self.obstacles:
            obs['x'] = obs['x0']
            obs['y'] = obs['y0']
            obs['t_redir'] = 0.0
            self._randomize_velocity(obs)

        self.ready = False
        self.pos_pub = self.create_publisher(Float32MultiArray, '/dynamic_obstacle_positions', 10)
        self.create_service(Empty, 'reset_obstacles', self._reset_cb)
        self.create_timer(5.0, self._on_ready, callback_group=None)
        self.create_timer(self.dt, self._update)
        

    def _on_ready(self):
        self.ready = True
        
    def _randomize_velocity(self, obs):
        angle = random.uniform(0, 2 * math.pi)
        obs['vx'] = self.speed * math.cos(angle)
        obs['vy'] = self.speed * math.sin(angle)
        obs['t_redir'] = 0.0

    def _set_pose(self, name, x, y):
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = 2          # MODEL = 2
        req.pose.position.x = x
        req.pose.position.y = y
        req.pose.position.z = 0.2
        req.pose.orientation.w = 1.0
        self.cli.call_async(req)     # fire-and-forget

    def _update(self):
        if not self.ready:
            return
        for obs in self.obstacles:
            obs['t_redir'] += self.dt

            nx = obs['x'] + obs['vx'] * self.dt
            ny = obs['y'] + obs['vy'] * self.dt

            # Obstacle bounces from the wall
            bounced = False
            if nx <= obs['xmin'] or nx >= obs['xmax']:
                obs['vx'] *= -1.0
                nx = max(obs['xmin'], min(obs['xmax'], nx))
                bounced = True
            if ny <= obs['ymin'] or ny >= obs['ymax']:
                obs['vy'] *= -1.0
                ny = max(obs['ymin'], min(obs['ymax'], ny))
                bounced = True

            # Redirect but not when it just bounced
            if obs['t_redir'] >= self.redir_interval and not bounced:
                self._randomize_velocity(obs)

            # If bounced, reset timer
            if bounced:
                obs['t_redir'] = 0.0

            obs['x'] = nx
            obs['y'] = ny
            self._set_pose(obs['name'], obs['x'], obs['y'])

        # Publish current positions for collision detection in environment.py
        msg = Float32MultiArray()
        msg.data = []
        for obs in self.obstacles:
            msg.data.extend([float(obs['x']), float(obs['y'])])
        self.pos_pub.publish(msg)

    def _reset_cb(self, _request, response):
        for obs in self.obstacles:
            obs['x'] = obs['x0']
            obs['y'] = obs['y0']
            self._randomize_velocity(obs)
            self._set_pose(obs['name'], obs['x'], obs['y'])
        return response


def main():
    rclpy.init()
    rclpy.spin(ObstacleMoverNode())
    rclpy.shutdown()