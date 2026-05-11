import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
import random, math


class ObstacleMoverNode(Node):
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
        self.redir_interval = 3.0

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
        self.get_logger().info(f'ObstacleMoverNode mode: {mode} ({len(self.obstacles)} obstacles)')

        for obs in self.obstacles:
            obs['x'] = obs['x0']
            obs['y'] = obs['y0']
            obs['t_redir'] = 0.0
            self._randomize_velocity(obs)

        self.ready = False
        self.create_service(Empty, 'reset_obstacles', self._reset_cb)
        self.create_timer(5.0, self._on_ready, callback_group=None)
        self.create_timer(self.dt, self._update)
        self.get_logger().info('ObstacleMoverNode ready — waiting 5s for Gazebo to load models')

    def _on_ready(self):
        self.ready = True
        self.get_logger().info('ObstacleMoverNode: starting obstacle movement')

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

            # 1. Najpierw odbicia od ścian
            bounced = False
            if nx <= obs['xmin'] or nx >= obs['xmax']:
                obs['vx'] *= -1.0
                nx = max(obs['xmin'], min(obs['xmax'], nx))
                bounced = True
            if ny <= obs['ymin'] or ny >= obs['ymax']:
                obs['vy'] *= -1.0
                ny = max(obs['ymin'], min(obs['ymax'], ny))
                bounced = True

            # 2. Potem redirect — ale nie jeśli właśnie odbiło
            if obs['t_redir'] >= self.redir_interval and not bounced:
                self._randomize_velocity(obs)

            # Jeśli odbiło, zresetuj timer żeby nie nakładało się zaraz potem
            if bounced:
                obs['t_redir'] = 0.0

            obs['x'] = nx
            obs['y'] = ny
            self._set_pose(obs['name'], obs['x'], obs['y'])

    def _reset_cb(self, _request, response):
        for obs in self.obstacles:
            obs['x'] = obs['x0']
            obs['y'] = obs['y0']
            self._randomize_velocity(obs)
            self._set_pose(obs['name'], obs['x'], obs['y'])
        self.get_logger().info('Obstacles reset')
        return response


def main():
    rclpy.init()
    rclpy.spin(ObstacleMoverNode())
    rclpy.shutdown()