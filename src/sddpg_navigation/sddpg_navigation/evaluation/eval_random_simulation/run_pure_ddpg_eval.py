import os
import pickle
import torch
import rclpy
from sddpg_navigation.evaluation.eval_random_simulation.rand_eval_gpu import RandEvalGpu
from sddpg_navigation.evaluation.eval_random_simulation.utility import gen_test_env_poly_list_env
from sddpg_navigation.training.train_pure_ddpg.ddpg_networks import ActorNet
import argparse

_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pure_ddpg_actor(net_dir, state_num=22, action_num=2, dim=(256, 256, 256), use_cuda=True):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    actor_net = ActorNet(state_num, action_num,
                         hidden1=dim[0], hidden2=dim[1], hidden3=dim[2])
    actor_net.load_state_dict(torch.load(net_dir, map_location='cpu'))
    actor_net.eval()
    return actor_net


def evaluate_pure_ddpg(pos_start=0, pos_end=199, model_name='pure_ddpg',
                       save_dir=os.path.join(_DIR, '..', 'saved_model') + os.sep,
                       state_num=22, is_scale=True, is_save_result=False, use_cuda=True):
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    eval_pos_path = os.path.join(_DIR, 'eval_positions.p')
    start_goal_pos = pickle.load(open(eval_pos_path, "rb"))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    net_dir = save_dir + model_name + '.pt'
    actor_net = load_pure_ddpg_actor(net_dir, state_num=state_num, use_cuda=use_cuda)
    eval = RandEvalGpu(actor_net, robot_init_list, goal_list, poly_list,
                       max_steps=2500, action_rand=0.01, goal_dis_min_dis=0.3,
                       is_scale=is_scale, is_poisson=False, use_cuda=use_cuda)
    data = eval.run_ros()
    if is_save_result:
        record_dir = os.path.join(_DIR, '..', 'record_data')
        os.makedirs(record_dir, exist_ok=True)
        pickle.dump(data,
                    open(os.path.join(record_dir, model_name + '_' + str(pos_start) + '_' + str(pos_end) + '.p'), 'wb+'))
    print(str(model_name) + " Pure DDPG Eval Finished ...")


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='pure_ddpg')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(_DIR, '..', 'saved_model') + os.sep)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    evaluate_pure_ddpg(use_cuda=USE_CUDA, model_name=args.model_name,
                       save_dir=args.save_dir, is_save_result=args.save == 1)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
