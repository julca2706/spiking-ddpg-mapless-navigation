import os
import pickle
import numpy as np
import torch
import rclpy
from sddpg_navigation.evaluation.eval_random_simulation.rand_eval_gpu import RandEvalGpu
from sddpg_navigation.evaluation.eval_random_simulation.utility import gen_test_env_poly_list_env
from sddpg_navigation.training.train_td3.td3_networks import ActorNet
from sddpg_navigation.utility import wheeled_network_2_robot_action_decoder
import argparse

_DIR = os.path.dirname(os.path.abspath(__file__))


def load_gru_actor_network(net_dir, state_num=22, action_num=2, dim=(256, 256, 256)):
    actor_net = ActorNet(state_num, action_num,
                         hidden1=dim[0], hidden2=dim[1], hidden3=dim[2],
                         last_action_num=action_num)
    actor_net.load_state_dict(torch.load(net_dir, map_location='cpu'))
    actor_net.eval()
    return actor_net


class GRURandEvalGpu(RandEvalGpu):
    def __init__(self, actor_net, *args, action_num=2, **kwargs):
        super().__init__(actor_net, *args, **kwargs)
        self._action_num = action_num
        self.gru_hidden = None
        self.gru_last_action = np.zeros(action_num)

    def _set_new_target(self, goal_ita):
        self.gru_hidden = None
        self.gru_last_action = np.zeros(self._action_num)
        super()._set_new_target(goal_ita)

    def _network_2_robot_action(self, state):
        with torch.no_grad():
            state_scaled = self._state_2_scale_state(state)
            state_tensor = torch.Tensor(
                np.array(state_scaled).reshape((1, 1, -1))
            ).to(self.device)
            last_action_tensor = torch.Tensor(
                self.gru_last_action.reshape((1, 1, -1))
            ).to(self.device)
            action, self.gru_hidden = self.actor_net(
                state_tensor, last_action=last_action_tensor, hidden=self.gru_hidden
            )
            action = action.to('cpu').numpy().squeeze()
        self.gru_last_action = np.array(action)
        noise = np.random.randn(self._action_num) * self.action_rand
        action = noise + (1 - self.action_rand) * action
        action = np.clip(action, [0., 0.], [1., 1.])
        action = wheeled_network_2_robot_action_decoder(action, self.max_spd, self.min_spd)
        return action


def evaluate_gru_ddpg(pos_start=0, pos_end=199, model_name='DDPG_R1_actor_network_s10',
                      save_dir=os.path.join(_DIR, '..', '..', 'save_td3_weights') + os.sep,
                      state_num=22, is_save_result=False, use_cuda=True):
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    eval_pos_path = os.path.join(_DIR, 'eval_positions.p')
    start_goal_pos = pickle.load(open(eval_pos_path, 'rb'))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    net_dir = save_dir + model_name + '.pt'
    actor_net = load_gru_actor_network(net_dir, state_num=state_num)
    eval_node = GRURandEvalGpu(actor_net, robot_init_list, goal_list, poly_list,
                               max_steps=2500, action_rand=0.01, goal_dis_min_dis=0.3,
                               is_scale=True, use_cuda=use_cuda)
    data = eval_node.run_ros()
    if is_save_result:
        record_dir = os.path.join(_DIR, '..', 'record_data')
        os.makedirs(record_dir, exist_ok=True)
        pickle.dump(data,
                    open(os.path.join(record_dir,
                                      model_name + '_' + str(pos_start) + '_' + str(pos_end) + '.p'), 'wb+'))
    print(model_name + ' GRU DDPG Eval Finished ...')


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='DDPG_R1_actor_network_s10')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(_DIR, '..', '..', 'save_td3_weights') + os.sep)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    evaluate_gru_ddpg(use_cuda=USE_CUDA, model_name=args.model_name,
                      save_dir=args.save_dir, is_save_result=args.save == 1)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
