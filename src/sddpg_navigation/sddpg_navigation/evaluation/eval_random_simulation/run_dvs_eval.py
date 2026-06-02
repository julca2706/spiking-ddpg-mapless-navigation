import os
import pickle
import numpy as np
import torch
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sddpg_navigation.evaluation.eval_random_simulation.rand_eval_gpu import RandEvalGpu
from sddpg_navigation.evaluation.eval_random_simulation.utility import gen_test_env_poly_list_env
from sddpg_navigation.training.train_td3_DVS.td3_dvs_networks import ActorNet
from sddpg_navigation.utility import wheeled_network_2_robot_action_decoder, ddpg_state_rescale
import argparse

_DIR = os.path.dirname(os.path.abspath(__file__))

# ros_gz_bridge publishes sensor topics with BEST_EFFORT reliability — must match exactly.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def load_dvs_actor(net_dir, action_num=2):
    """Load DVS ActorNet (CNN+GRU) from a .pt checkpoint file."""
    actor_net = ActorNet(4, action_num, last_action_num=action_num)
    actor_net.load_state_dict(torch.load(net_dir, map_location='cpu'))
    actor_net.eval()
    return actor_net


class DVSRandEvalGpu(RandEvalGpu):
    """Evaluation node for the DVS (CNN+GRU) actor.

    Extends RandEvalGpu with:
    - /camera/events subscription (BEST_EFFORT QoS) that decodes mono8 frames into
      two float32 channels: ON (raw > 200) and OFF (raw < 50).
    - GRU hidden state and last_action maintained across steps, reset each episode.
    - _network_2_robot_action() using event frame (2,64,64) + goal state[:2] + last_action.

    Requires evaluation_dynamic.launch.py (includes event_camera and obstacle_mover nodes).
    """

    def __init__(self, actor_net, *args, action_num=2, **kwargs):
        self._bridge = CvBridge()
        self._event_frame_init = False
        self._event_frame = np.zeros((2, 64, 64), dtype=np.float32)
        self._action_num = action_num
        self._gru_hidden = None
        self._last_action = np.zeros(action_num, dtype=np.float32)

        # Base class creates the ROS node, subscribes to odom/scan/pose, moves actor to device.
        super().__init__(actor_net, *args, **kwargs)

        # Add event camera subscription after the node exists.
        self.create_subscription(Image, '/camera/events', self._events_cb, _SENSOR_QOS)

        self.get_logger().info("Waiting for /camera/events...")
        while not self._event_frame_init:
            rclpy.spin_once(self, timeout_sec=1.0)
        self.get_logger().info("Event frames ready.")

    def _events_cb(self, msg):
        """Decode mono8 event frame: ON=255 → ch0, OFF=0 → ch1, neutral=128 ignored."""
        raw = self._bridge.imgmsg_to_cv2(msg, 'mono8')
        on_ch  = (raw > 200).astype(np.float32)
        off_ch = (raw < 50).astype(np.float32)
        self._event_frame = np.stack([on_ch, off_ch], axis=0)
        if not self._event_frame_init:
            self._event_frame_init = True

    def _set_new_target(self, goal_ita):
        """Reset GRU hidden state and last_action at the start of each episode."""
        self._gru_hidden = None
        self._last_action = np.zeros(self._action_num, dtype=np.float32)
        super()._set_new_target(goal_ita)

    def _network_2_robot_action(self, state):
        """Run DVS actor: event frame (2,64,64) + goal state[:2] + last_action → wheel speeds.

        state[:2] = [goal_dir, goal_dis] — same values used during training.
        Event frame is taken from the latest /camera/events message.
        """
        with torch.no_grad():
            # (1, 1, 2, 64, 64) — batch=1, seq_len=1
            events_t = torch.FloatTensor(self._event_frame).unsqueeze(0).unsqueeze(0).to(self.device)
            rescale = ddpg_state_rescale(state, len(state))
            goal_t = torch.FloatTensor(
                np.array(rescale[:4], dtype=np.float32)
            ).reshape(1, 1, 4).to(self.device)
            last_action_t = torch.FloatTensor(
                self._last_action
            ).reshape(1, 1, -1).to(self.device)

            action, self._gru_hidden = self.actor_net(
                events_t, goal_t,
                last_action=last_action_t,
                hidden=self._gru_hidden
            )
            action = action.cpu().numpy().squeeze()

        self._last_action = action.astype(np.float32)
        noise = np.random.randn(self._action_num) * self.action_rand
        action = np.clip(noise + (1.0 - self.action_rand) * action, 0.0, 1.0)
        action = wheeled_network_2_robot_action_decoder(action, self.max_spd, self.min_spd)
        return action


def evaluate_dvs(pos_start=0, pos_end=199, model_name='DVS_R1',
                 save_dir=os.path.join(_DIR, '..', '..', 'save_dvs_weights') + os.sep,
                 checkpoint=None, is_save_result=False, use_cuda=True):
    """
    Evaluate DVS agent in simulation.

    :param model_name: training run name (prefix used when saving weights)
    :param save_dir: directory containing saved actor weights
    :param checkpoint: step index of the checkpoint to load
                       (loads {model_name}_dvs_actor_s{checkpoint}.pt);
                       if None, loads {model_name}_dvs_actor_s0.pt (final save)
    :param is_save_result: if True, save evaluation results to record_data/
    :param use_cuda: if True, use GPU
    """
    poly_list, _ = gen_test_env_poly_list_env()
    eval_pos_path = os.path.join(_DIR, 'eval_positions.p')
    start_goal_pos = pickle.load(open(eval_pos_path, 'rb'))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list       = start_goal_pos[1][pos_start:pos_end + 1]

    step = checkpoint if checkpoint is not None else 0
    net_dir = os.path.join(save_dir, f'{model_name}_dvs_actor_s{step}.pt')
    actor_net = load_dvs_actor(net_dir)

    eval_node = DVSRandEvalGpu(
        actor_net, robot_init_list, goal_list, poly_list,
        max_steps=2500, action_rand=0.01, goal_dis_min_dis=0.3,
        is_scale=True, use_cuda=use_cuda
    )
    data = eval_node.run_ros()

    if is_save_result:
        record_dir = os.path.join(_DIR, '..', 'record_data')
        os.makedirs(record_dir, exist_ok=True)
        out_path = os.path.join(record_dir,
                                f'{model_name}_{pos_start}_{pos_end}.p')
        pickle.dump(data, open(out_path, 'wb+'))

    print(f'{model_name} DVS Eval Finished ...')


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='DVS_R1')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(_DIR, '..', '..', 'save_dvs_weights') + os.sep)
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Step index of checkpoint to load '
                             '({model_name}_dvs_actor_s{checkpoint}.pt). '
                             'Defaults to 0 (final save).')
    args = parser.parse_args()

    evaluate_dvs(
        use_cuda=args.cuda == 1,
        model_name=args.model_name,
        save_dir=args.save_dir,
        checkpoint=args.checkpoint,
        is_save_result=args.save == 1
    )
    rclpy.shutdown()


if __name__ == '__main__':
    main()
