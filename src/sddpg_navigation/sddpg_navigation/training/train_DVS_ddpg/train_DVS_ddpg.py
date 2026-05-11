import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sddpg_navigation.training.train_DVS_ddpg.ddpg_DVS_agent import AgentDVS
from sddpg_navigation.environment import *
from sddpg_navigation.utility import *

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class EventSubscriber(Node):
    """Minimal node that buffers the latest decoded event frame from /camera/events."""

    def __init__(self):
        super().__init__('dvs_event_subscriber')
        self.bridge = CvBridge()
        self.robot_events_init = False
        self.robot_events = np.zeros((2, 64, 64), dtype=np.float32)
        self.create_subscription(Image, '/camera/events', self._robot_events_cb, _SENSOR_QOS)

    def _robot_events_cb(self, msg):
        if not self.robot_events_init:
            self.robot_events_init = True
        raw = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        on_ch  = (raw > 200).astype(np.float32)
        off_ch = (raw < 50).astype(np.float32)
        self.robot_events = np.stack([on_ch, off_ch], axis=0)

    def get_decoded(self):
        """Return (2, 64, 64) float32: channel 0 = ON, channel 1 = OFF."""
        return self.robot_events.copy()


def train_dvs_ddpg(run_name="DVS_R1", exp_name="Rand_R1",
                   episode_num=(100, 200, 300, 400),
                   iteration_num_start=(200, 300, 400, 500),
                   iteration_num_step=(1, 2, 3, 4),
                   iteration_num_max=(1000, 1000, 1000, 1000),
                   linear_spd_max=0.5, linear_spd_min=0.05, save_steps=10000,
                   env_epsilon=(0.9, 0.6, 0.6, 0.6),
                   env_epsilon_decay=(0.999, 0.9999, 0.9999, 0.9999),
                   laser_half_num=9, laser_min_dis=0.35, scan_overall_num=36,
                   goal_dis_min_dis=0.3, obs_reward=-20, goal_reward=30,
                   goal_dis_amp=15, goal_th=0.5, obs_th=0.35,
                   state_num=22, action_num=2,
                   memory_size=5000, batch_size=128, epsilon_end=0.1,
                   rand_start=10000, rand_decay=0.999, rand_step=2,
                   target_tau=0.01, target_step=5, start_env=0, use_cuda=True):

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_dir = 'save_dvs_weights'
    try:
        os.mkdir(os.path.join(base_dir, weights_dir))
        print("Directory ", weights_dir, " Created ")
    except FileExistsError:
        print("Directory ", weights_dir, " already exists")

    # Define 4 training environments
    env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(episode_num[0])
    env2_poly_list, env2_raw_poly_list, env2_goal_list, env2_init_list = gen_rand_list_env2(episode_num[1])
    env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(episode_num[2])
    env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(episode_num[3])
    overall_poly_list = [env1_poly_list, env2_poly_list, env3_poly_list, env4_poly_list]

    positions_dir = 'random_positions'
    overall_list = pickle.load(open(os.path.join(base_dir, positions_dir, exp_name + ".p"), "rb"))
    overall_init_list = overall_list[0]
    overall_goal_list = overall_list[1]
    print("Use Training Rand Start and Goal Positions: ", exp_name)

    env = GazeboEnvironment(laser_scan_half_num=laser_half_num,
                            laser_scan_min_dis=laser_min_dis,
                            scan_dir_num=scan_overall_num,
                            goal_dis_min_dis=goal_dis_min_dis,
                            obs_reward=obs_reward, goal_reward=goal_reward,
                            goal_dis_amp=goal_dis_amp,
                            goal_near_th=goal_th, obs_near_th=obs_th)

    event_sub = EventSubscriber()

    agent = AgentDVS(state_num, action_num,
                     memory_size=memory_size, batch_size=batch_size,
                     epsilon_end=epsilon_end,
                     epsilon_rand_decay_start=rand_start,
                     epsilon_decay=rand_decay,
                     epsilon_rand_decay_step=rand_step,
                     target_tau=target_tau,
                     target_update_steps=target_step,
                     use_cuda=use_cuda)

    tb_writer = SummaryWriter()

    overall_steps = 0
    overall_episode = 0
    env_episode = 0
    env_ita = start_env
    ita_per_episode = iteration_num_start[env_ita]
    env.set_new_environment(overall_init_list[env_ita],
                            overall_goal_list[env_ita],
                            overall_poly_list[env_ita])
    agent.reset_epsilon(env_epsilon[env_ita], env_epsilon_decay[env_ita])

    start_time = time.time()
    while True:
        state = env.reset(env_episode)
        # Get initial event frame after reset
        rclpy.spin_once(event_sub, timeout_sec=0.05)
        event_frame = event_sub.get_decoded()
        episode_reward = 0

        for ita in range(ita_per_episode):
            ita_time_start = time.time()
            overall_steps += 1

            raw_action = agent.act(event_frame, state)
            decode_action = wheeled_network_2_robot_action_decoder(
                raw_action, linear_spd_max, linear_spd_min
            )

            next_state, reward, done = env.step(decode_action)

            # Get new event frame after step
            rclpy.spin_once(event_sub, timeout_sec=0.05)
            next_event_frame = event_sub.get_decoded()

            episode_reward += reward
            agent.remember_seq(state, event_frame, raw_action, reward,
                                next_state, next_event_frame, done)
            state = next_state
            event_frame = next_event_frame

            if len(agent.memory) > batch_size:
                actor_loss_value, critic_loss_value = agent.replay()
                tb_writer.add_scalar('DVS/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('DVS/critic_loss', critic_loss_value, overall_steps)

            ita_time_end = time.time()
            tb_writer.add_scalar('DVS/ita_time', ita_time_end - ita_time_start, overall_steps)
            tb_writer.add_scalar('DVS/action_epsilon', agent.epsilon, overall_steps)
            tb_writer.add_scalar('DVS/raw_left_wheel_output', raw_action[0], overall_steps)
            tb_writer.add_scalar('DVS/raw_right_wheel_output', raw_action[1], overall_steps)

            if overall_steps % save_steps == 0:
                agent.save(os.path.join(base_dir, weights_dir),
                           overall_steps // save_steps, run_name)

            if done or ita == ita_per_episode - 1:
                print("Episode: {}/{}, Avg Reward: {}, Steps: {}".format(
                    overall_episode, episode_num, episode_reward / (ita + 1), ita + 1))
                tb_writer.add_scalar('DVS/avg_reward', episode_reward / (ita + 1), overall_steps)
                break

        if ita_per_episode < iteration_num_max[env_ita]:
            ita_per_episode += iteration_num_step[env_ita]
        if overall_episode == 999:
            agent.save(os.path.join(base_dir, weights_dir), 0, run_name)
        overall_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 3:
                break
            env_ita += 1
            env.set_new_environment(overall_init_list[env_ita],
                                    overall_goal_list[env_ita],
                                    overall_poly_list[env_ita])
            agent.reset_epsilon(env_epsilon[env_ita], env_epsilon_decay[env_ita])
            ita_per_episode = iteration_num_start[env_ita]
            env_episode = 0

    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--start_env', type=int, default=0)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    train_dvs_ddpg(use_cuda=USE_CUDA, start_env=args.start_env)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
