import os
import rclpy
from sddpg_navigation.evaluation.eval_random_simulation.rand_eval_gpu import RandEvalGpu
from sddpg_navigation.evaluation.eval_random_simulation.utility import *
import argparse

_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_sddpg(pos_start=0, pos_end=199, model_name='sddpg_bw_5',
                   save_dir=os.path.join(_DIR, '..', 'saved_model') + os.sep,
                   batch_window=5, is_save_result=False, use_cuda=True,
                   checkpoint=None):
    """
    Evaluate Spiking DDPG in Simulated Environment

    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model (run_name prefix used during training)
    :param save_dir: directory to the saved model
    :param batch_window: inference timesteps
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    :param checkpoint: episode number of the checkpoint to load (uses training naming
                       format: {model_name}_snn_weights_s{checkpoint}.p); if None,
                       uses legacy format: {model_name}_weights.p
    """
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    eval_pos_path = os.path.join(_DIR, 'eval_positions.p')
    start_goal_pos = pickle.load(open(eval_pos_path, "rb"))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    if checkpoint is not None:
        # Training saves as: {run_name}_snn_weights_s{episode}.p
        w_dir = os.path.join(save_dir, model_name + '_snn_weights_s' + str(checkpoint) + '.p')
        b_dir = os.path.join(save_dir, model_name + '_snn_bias_s' + str(checkpoint) + '.p')
    else:
        # Legacy format: {model_name}_weights.p
        w_dir = save_dir + model_name + '_weights.p'
        b_dir = save_dir + model_name + '_bias.p'
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    actor_net = load_test_actor_snn_network(w_dir, b_dir, device, batch_window=batch_window)
    eval = RandEvalGpu(actor_net, robot_init_list, goal_list, poly_list,
                       max_steps=1000, action_rand=0.01, goal_dis_min_dis=0.3,
                       is_spike=True, use_cuda=use_cuda, batch_window=batch_window)
    data = eval.run_ros()
    if is_save_result:
        record_dir = os.path.join(_DIR, '..', 'record_data')
        os.makedirs(record_dir, exist_ok=True)
        pickle.dump(data,
                    open(os.path.join(record_dir, model_name + '_' + str(pos_start) + '_' + str(pos_end) + '.p'), 'wb+'))
    print(str(model_name) + " Eval on GPU Finished ...")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--step', type=int, default=5)
    # New args to allow evaluating a custom model from a different directory
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    # checkpoint: episode number from training (uses training naming convention)
    parser.add_argument('--checkpoint', type=int, default=None)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    # Default: legacy model; override with --model_name / --save_dir / --checkpoint
    # MODEL_NAME = 'sddpg_bw_' + str(args.step)
    MODEL_NAME = args.model_name if args.model_name else 'sddpg_bw_' + str(args.step)
    kwargs = dict(use_cuda=USE_CUDA, model_name=MODEL_NAME,
                  batch_window=args.step, is_save_result=SAVE_RESULT,
                  checkpoint=args.checkpoint)
    if args.save_dir:
        kwargs['save_dir'] = args.save_dir
    evaluate_sddpg(**kwargs)
    rclpy.shutdown()

if __name__ == '__main__':
    main()