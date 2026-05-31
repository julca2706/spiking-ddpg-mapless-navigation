from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import os
from sddpg_navigation.training.train_td3_DVS.td3_dvs_networks import ActorNet, CriticNet

STATE_NUM = 4  # goal_dir, goal_dis, odom_linear, odom_angular from rescale_state[:4]


class Agent:
    def __init__(self,
                 state_num,
                 action_num,
                 rescale_state_num,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 memory_size=100000,
                 batch_size=128,
                 target_tau=0.005,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 critic_lr=1e-4,
                 actor_lr=1e-4,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=60000,
                 epsilon_rand_decay_step=1,
                 policy_delay=2,
                 seq_len=10,
                 use_cuda=True):
        self.state_num = state_num
        self.action_num = action_num
        self.rescale_state_num = rescale_state_num
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.reward_gamma = reward_gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_rand_decay_start = epsilon_rand_decay_start
        self.epsilon_rand_decay_step = epsilon_rand_decay_step
        self.policy_delay = policy_delay
        self.seq_len = seq_len
        self.use_cuda = use_cuda
        self.epsilon = epsilon_start

        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.memory = deque(maxlen=self.memory_size)

        # Actor: state[:4] (no LiDAR) + CNN events; critic: full LiDAR state
        self.actor_net = ActorNet(STATE_NUM, self.action_num,
                                  hidden3=actor_net_dim[2],
                                  last_action_num=self.action_num)
        self.target_actor_net = ActorNet(STATE_NUM, self.action_num,
                                         hidden3=actor_net_dim[2],
                                         last_action_num=self.action_num)
        self.critic_net = CriticNet(self.state_num, self.action_num,
                                    hidden1=critic_net_dim[0],
                                    hidden2=critic_net_dim[1],
                                    hidden3=critic_net_dim[2])
        self.critic_net2 = CriticNet(self.state_num, self.action_num,
                                     hidden1=critic_net_dim[0],
                                     hidden2=critic_net_dim[1],
                                     hidden3=critic_net_dim[2])
        self.target_critic_net = CriticNet(self.state_num, self.action_num,
                                           hidden1=critic_net_dim[0],
                                           hidden2=critic_net_dim[1],
                                           hidden3=critic_net_dim[2])
        self.target_critic_net2 = CriticNet(self.state_num, self.action_num,
                                            hidden1=critic_net_dim[0],
                                            hidden2=critic_net_dim[1],
                                            hidden3=critic_net_dim[2])
        self._hard_update(self.target_actor_net, self.actor_net)
        self._hard_update(self.target_critic_net, self.critic_net)
        self._hard_update(self.target_critic_net2, self.critic_net2)
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.critic_net2.to(self.device)
        self.target_actor_net.to(self.device)
        self.target_critic_net.to(self.device)
        self.target_critic_net2.to(self.device)

        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.critic_lr)

        self.step_ita = 0
        self.last_actor_loss = 0.0
        self.current_seq = []
        self.hidden_state = None
        self.last_action = np.zeros(self.action_num)
        self.prev_last_action = np.zeros(self.action_num)
        self.prev_hidden_state = None
        self.seq_start_hidden = None
        self.seq_start_last_action = np.zeros(self.action_num)

    def remember_seq(self, state, rescale_state, event_frame,
                     action, reward,
                     next_state, rescale_next_state, next_event_frame, done):
        if len(self.current_seq) == 0:
            self.seq_start_hidden = self.prev_hidden_state
            self.seq_start_last_action = self.prev_last_action.copy()

        self.current_seq.append((state, rescale_state, event_frame,
                                  action, reward,
                                  next_state, rescale_next_state, next_event_frame, done))

        if len(self.current_seq) == self.seq_len or done:
            seq = self.current_seq
            pad_len = self.seq_len - len(seq)
            actions_in_seq = [t[3] for t in seq]
            seq_last_actions_real = [self.seq_start_last_action] + actions_in_seq[:-1]

            if pad_len > 0:
                zero_rstate = np.zeros_like(seq[0][1])
                zero_ev = np.zeros((2, 64, 64), dtype=np.float32)
                padding = [(np.zeros_like(seq[0][0]), zero_rstate, zero_ev,
                            np.zeros_like(seq[0][3]), 0,
                            np.zeros_like(seq[0][5]), zero_rstate, zero_ev, False)] * pad_len
                seq = padding + seq
                seq_last_actions = [np.zeros(self.action_num)] * pad_len + seq_last_actions_real
            else:
                seq_last_actions = seq_last_actions_real

            seq_rescale_states = np.array([t[1] for t in seq])
            seq_rescale_next_states = np.array([t[6] for t in seq])
            seq_last_actions = np.array(seq_last_actions)
            seq_states = np.array([t[0] for t in seq])
            seq_nstates = np.array([t[5] for t in seq])
            seq_rewards = np.array([t[4] for t in seq], dtype=np.float32)
            seq_dones = np.array([t[8] for t in seq], dtype=np.float32)
            seq_events = np.stack([t[2] for t in seq], axis=0)       # (T, 2, 64, 64)
            seq_next_events = np.stack([t[7] for t in seq], axis=0)  # (T, 2, 64, 64)

            hidden_size = self.actor_net.gru.hidden_size
            if self.seq_start_hidden is not None:
                init_hidden = self.seq_start_hidden.detach().cpu().numpy()
            else:
                init_hidden = np.zeros((1, 1, hidden_size))

            last = self.current_seq[-1]
            self.memory.append((seq_rescale_states, seq_rescale_next_states, seq_last_actions,
                                 last[0], last[3], last[4], last[5], last[8], init_hidden,
                                 seq_states, seq_nstates, seq_rewards, seq_dones,
                                 seq_events, seq_next_events))
            self.current_seq = []

        if done:
            self.hidden_state = None
            self.last_action = np.zeros(self.action_num)

    def act(self, state, event_frame, explore=True, train=True):
        with torch.no_grad():
            events_t = torch.FloatTensor(event_frame).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,2,64,64)
            state_t = torch.FloatTensor(np.array(state[:STATE_NUM], dtype=np.float32)).reshape(1, 1, STATE_NUM).to(self.device)
            last_action_t = torch.FloatTensor(self.last_action.reshape(1, 1, -1)).to(self.device)
            self.prev_hidden_state = self.hidden_state
            action, self.hidden_state = self.actor_net(events_t, state_t,
                                                       last_action=last_action_t,
                                                       hidden=self.hidden_state)
            action = action.to('cpu').numpy().squeeze()

        if train:
            if self.step_ita > self.epsilon_rand_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_rand_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.randn(self.action_num) * self.epsilon
            action = noise + (1 - self.epsilon) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        elif explore:
            noise = np.random.randn(self.action_num) * self.epsilon_end
            action = noise + (1 - self.epsilon_end) * action
            action = np.clip(action, [0., 0.], [1., 1.])

        self.prev_last_action = self.last_action.copy()
        self.last_action = np.array(action)
        return action.tolist()

    def replay(self):
        (seq_rstate_batch, seq_rnstate_batch, seq_last_action_batch,
         state_batch, action_batch, reward_batch, nstate_batch, done_batch, hidden_batch,
         seq_states_batch, seq_nstates_batch, seq_rewards_batch, seq_dones_batch,
         seq_ev_batch, seq_nev_batch) = self._random_minibatch()
        B, T = self.batch_size, self.seq_len

        # goal slices: (B, T, STATE_NUM)
        seq_state_batch = seq_rstate_batch[:, :, :STATE_NUM]
        seq_nstate_batch = seq_rnstate_batch[:, :, :STATE_NUM]

        with torch.no_grad():
            next_seq_last_actions = torch.cat([
                seq_last_action_batch[:, 1:, :],
                action_batch.unsqueeze(1)
            ], dim=1)
            _, hidden_out_batch = self.target_actor_net(seq_ev_batch, seq_state_batch,
                                                        last_action=seq_last_action_batch,
                                                        hidden=hidden_batch)
            all_nactions, _ = self.target_actor_net(seq_nev_batch, seq_nstate_batch,
                                                    last_action=next_seq_last_actions,
                                                    hidden=hidden_out_batch,
                                                    return_seq=True)  # (B, T, action_num)
            noise = torch.clamp(torch.randn_like(all_nactions) * 0.1, -0.2, 0.2)
            all_nactions = torch.clamp(all_nactions + noise, 0.0, 1.0)
            all_nstates_flat = seq_nstates_batch.view(B * T, -1)
            all_nactions_flat = all_nactions.view(B * T, -1)
            tq1_all = self.target_critic_net([all_nstates_flat, all_nactions_flat])
            tq2_all = self.target_critic_net2([all_nstates_flat, all_nactions_flat])
            all_target_q = (seq_rewards_batch.view(B * T, 1) +
                            self.reward_gamma * torch.min(tq1_all, tq2_all) *
                            (1. - seq_dones_batch.view(B * T, 1)))

        all_states_flat = seq_states_batch.view(B * T, -1)
        all_actions_flat = next_seq_last_actions.view(B * T, -1)

        self.critic_optimizer.zero_grad()
        current_q_all = self.critic_net([all_states_flat, all_actions_flat])
        critic_loss = self.criterion(current_q_all, all_target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.critic_optimizer2.zero_grad()
        current_q2_all = self.critic_net2([all_states_flat, all_actions_flat])
        critic_loss2 = self.criterion(current_q2_all, all_target_q)
        critic_loss2.backward()
        nn.utils.clip_grad_norm_(self.critic_net2.parameters(), max_norm=0.5)
        self.critic_optimizer2.step()

        self.step_ita += 1
        if self.step_ita % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            current_actions, _ = self.actor_net(seq_ev_batch, seq_state_batch,
                                                 last_action=seq_last_action_batch,
                                                 hidden=hidden_batch,
                                                 return_seq=True)  # (B, T, action_num)
            actor_loss = -self.critic_net([all_states_flat, current_actions.view(B * T, -1)])
            actor_loss = actor_loss.mean()
            self.last_actor_loss = actor_loss.item()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            if self.step_ita % self.target_update_steps == 0:
                self._soft_update(self.target_actor_net, self.actor_net)
                self._soft_update(self.target_critic_net, self.critic_net)
                self._soft_update(self.target_critic_net2, self.critic_net2)

        return self.last_actor_loss, critic_loss_item

    def reset_epsilon(self, new_epsilon, new_decay):
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay

    def save(self, save_dir, episode, run_name):
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        torch.save(self.actor_net.state_dict(),
                   save_dir + '/' + run_name + '_actor_network_s' + str(episode) + '.pt')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, load_file_name):
        self.actor_net.to('cpu')
        self.actor_net.load_state_dict(torch.load(load_file_name))
        self.actor_net.to(self.device)

    def _random_minibatch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        hidden_size = self.actor_net.gru.hidden_size
        B, T = self.batch_size, self.seq_len

        seq_rstate_batch   = np.zeros((B, T, self.rescale_state_num))
        seq_rnstate_batch  = np.zeros((B, T, self.rescale_state_num))
        seq_last_action_batch = np.zeros((B, T, self.action_num))
        state_batch        = np.zeros((B, self.state_num))
        action_batch       = np.zeros((B, self.action_num))
        reward_batch       = np.zeros((B, 1))
        nstate_batch       = np.zeros((B, self.state_num))
        done_batch         = np.zeros((B, 1))
        hidden_batch       = np.zeros((1, B, hidden_size))
        seq_states_batch   = np.zeros((B, T, self.state_num))
        seq_nstates_batch  = np.zeros((B, T, self.state_num))
        seq_rewards_batch  = np.zeros((B, T, 1))
        seq_dones_batch    = np.zeros((B, T, 1))
        seq_ev_batch       = np.zeros((B, T, 2, 64, 64), dtype=np.float32)
        seq_nev_batch      = np.zeros((B, T, 2, 64, 64), dtype=np.float32)

        for num in range(B):
            (seq_rstate_batch[num], seq_rnstate_batch[num], seq_last_action_batch[num],
             state_batch[num], action_batch[num], reward_batch[num, 0],
             nstate_batch[num], done_batch[num, 0], hidden_arr,
             seq_states_batch[num], seq_nstates_batch[num],
             seq_rewards_batch[num, :, 0], seq_dones_batch[num, :, 0],
             seq_ev_batch[num], seq_nev_batch[num]) = minibatch[num]
            hidden_batch[:, num, :] = hidden_arr[:, 0, :]

        def t(arr):
            return torch.Tensor(arr).to(self.device)

        return (t(seq_rstate_batch), t(seq_rnstate_batch), t(seq_last_action_batch),
                t(state_batch), t(action_batch), t(reward_batch),
                t(nstate_batch), t(done_batch), t(hidden_batch),
                t(seq_states_batch), t(seq_nstates_batch), t(seq_rewards_batch), t(seq_dones_batch),
                t(seq_ev_batch), t(seq_nev_batch))

    def _hard_update(self, target, source):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.data.copy_(p.data)

    def _soft_update(self, target, source):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.target_tau) + p.data * self.target_tau)
