from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import os
from sddpg_navigation.training.train_DVS_ddpg.ddpg_DVS_networks import ActorNet, CriticNet



class AgentDVS:
    """
    TD3 Agent with DVS actor (CNN+GRU) and LiDAR twin critics.

    Actor input : event frames np.float32 (2, 64, 64) already decoded + goal = state[:2]
    Critic input: LiDAR state (state_num dim)
    """
    def __init__(self,
                 state_num,
                 action_num,
                 critic_net_dim=(512, 512, 512),
                 memory_size=500,
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 critic_lr=1e-4,
                 actor_lr=1e-4,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=10000,
                 epsilon_rand_decay_step=1,
                 policy_delay=2,
                 seq_len=10,
                 use_cuda=True):
        self.state_num = state_num
        self.action_num = action_num
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

        self.actor_net = ActorNet(self.action_num)
        self.target_actor_net = ActorNet(self.action_num)
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
        self.target_actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.critic_net2.to(self.device)
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
        self.prev_hidden_state = None
        self.last_action = np.zeros(self.action_num, dtype=np.float32)
        self.prev_last_action = np.zeros(self.action_num, dtype=np.float32)
        self.seq_start_hidden = None
        self.seq_start_last_action = np.zeros(self.action_num, dtype=np.float32)

    def remember_seq(self, state, rescale_state, event_frame, action, reward,
                     next_state, rescale_next_state, next_event_frame, done):
        """
        Accumulate transitions and store sequences of length seq_len.

        state / next_state               : array (state_num,) — raw LiDAR state for critic
        rescale_state / rescale_next_state: array (state_num,) — rescaled state; goal = [:2]
        event_frame / next_event_frame   : np.float32 (2, 64, 64)
        action                           : array (action_num,)
        reward                           : float
        done                             : bool
        """
        if len(self.current_seq) == 0:
            self.seq_start_hidden = self.prev_hidden_state
            self.seq_start_last_action = self.prev_last_action.copy()

        # tuple: 0=state, 1=rescale_state, 2=event_frame, 3=action, 4=reward,
        #        5=next_state, 6=rescale_next_state, 7=next_event_frame, 8=done
        self.current_seq.append((np.array(state, dtype=np.float32),
                                  np.array(rescale_state, dtype=np.float32),
                                  np.array(event_frame, dtype=np.float32),
                                  np.array(action, dtype=np.float32),
                                  float(reward),
                                  np.array(next_state, dtype=np.float32),
                                  np.array(rescale_next_state, dtype=np.float32),
                                  np.array(next_event_frame, dtype=np.float32),
                                  bool(done)))

        if len(self.current_seq) == self.seq_len or done:
            seq = self.current_seq
            pad_len = self.seq_len - len(seq)

            actions_in_seq = [t[3] for t in seq]
            seq_last_actions_real = [self.seq_start_last_action] + actions_in_seq[:-1]

            if pad_len > 0:
                zero_ev    = np.zeros((2, 64, 64), dtype=np.float32)
                zero_state = np.zeros(self.state_num, dtype=np.float32)
                zero_act   = np.zeros(self.action_num, dtype=np.float32)
                padding = [(zero_state, zero_state, zero_ev, zero_act, 0.0,
                            zero_state, zero_state, zero_ev, False)] * pad_len
                seq = padding + seq
                seq_last_actions = ([np.zeros(self.action_num, dtype=np.float32)] * pad_len
                                    + seq_last_actions_real)
            else:
                seq_last_actions = seq_last_actions_real

            seq_events      = np.stack([t[2] for t in seq], axis=0)       # (T, 2, 64, 64)
            seq_goals       = np.stack([t[1][:4] for t in seq], axis=0)   # (T, 4) rescaled
            seq_next_events = np.stack([t[7] for t in seq], axis=0)       # (T, 2, 64, 64)
            seq_next_goals  = np.stack([t[6][:4] for t in seq], axis=0)   # (T, 4) rescaled
            seq_last_actions = np.stack(seq_last_actions, axis=0)         # (T, action_num)
            seq_states  = np.stack([t[0] for t in seq], axis=0)           # (T, state_num) raw
            seq_nstates = np.stack([t[5] for t in seq], axis=0)           # (T, state_num) raw
            seq_rewards = np.array([t[4] for t in seq], dtype=np.float32) # (T,)
            seq_dones   = np.array([t[8] for t in seq], dtype=np.float32) # (T,)

            hidden_size = self.actor_net.gru.hidden_size
            if self.seq_start_hidden is not None:
                init_hidden = self.seq_start_hidden.detach().cpu().numpy()
            else:
                init_hidden = np.zeros((1, 1, hidden_size), dtype=np.float32)

            last = self.current_seq[-1]
            self.memory.append((seq_events, seq_goals, seq_next_events, seq_next_goals,
                                 seq_last_actions,
                                 last[0], last[3], last[4], last[5], last[8],
                                 init_hidden,
                                 seq_states, seq_nstates, seq_rewards, seq_dones))
            self.current_seq = []

        if done:
            self.hidden_state = None
            self.last_action = np.zeros(self.action_num, dtype=np.float32)

    def act(self, event_frame, state, explore=True, train=True):
        """
        Generate action from event frame and state.

        event_frame : np.float32 (2, 64, 64) — decoded ON/OFF event channels
        state       : array (state_num,) — rescaled state; goal extracted as state[:2]
        """
        with torch.no_grad():
            events_t = torch.FloatTensor(event_frame).unsqueeze(0).unsqueeze(0).to(self.device)
            goal_t = torch.FloatTensor(np.array(state[:4], dtype=np.float32)).reshape(1, 1, 4).to(self.device)
            last_action_t = torch.FloatTensor(self.last_action).reshape(1, 1, -1).to(self.device)
            self.prev_hidden_state = self.hidden_state
            action, self.hidden_state = self.actor_net(events_t, goal_t,
                                                       last_action=last_action_t,
                                                       hidden=self.hidden_state)
            action = action.cpu().numpy().squeeze()

        if train:
            if self.step_ita > self.epsilon_rand_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_rand_decay_step == 0:
                    self.epsilon *= self.epsilon_decay
            noise = np.random.randn(self.action_num) * self.epsilon
            action = np.clip(noise + (1.0 - self.epsilon) * action, 0.0, 1.0)
        elif explore:
            noise = np.random.randn(self.action_num) * self.epsilon_end
            action = np.clip(noise + (1.0 - self.epsilon_end) * action, 0.0, 1.0)

        self.prev_last_action = self.last_action.copy()
        self.last_action = action.astype(np.float32)
        return action.tolist()

    def replay(self):
        """Experience Replay — one TD3 update step."""
        (seq_ev, seq_goal, seq_nev, seq_ngoal,
         seq_la, state_b, action_b, reward_b,
         nstate_b, done_b, hidden_b,
         seq_states_b, seq_nstates_b, seq_rewards_b, seq_dones_b) = self._random_minibatch()
        B, T = self.batch_size, self.seq_len

        # ── Target Q (all T steps) ────────────────────────────────────────
        with torch.no_grad():
            next_seq_la = torch.cat([seq_la[:, 1:, :], action_b.unsqueeze(1)], dim=1)
            _, hidden_out_b = self.target_actor_net(seq_ev, seq_goal,
                                                    last_action=seq_la, hidden=hidden_b)
            all_nactions, _ = self.target_actor_net(seq_nev, seq_ngoal,
                                                     last_action=next_seq_la,
                                                     hidden=hidden_out_b,
                                                     return_seq=True)  # (B, T, action_num)
            noise = torch.clamp(torch.randn_like(all_nactions) * 0.1, -0.2, 0.2)
            all_nactions = torch.clamp(all_nactions + noise, 0.0, 1.0)
            all_nstates_flat = seq_nstates_b.view(B * T, -1)
            all_nactions_flat = all_nactions.view(B * T, -1)
            tq1 = self.target_critic_net([all_nstates_flat, all_nactions_flat])
            tq2 = self.target_critic_net2([all_nstates_flat, all_nactions_flat])
            all_target_q = (seq_rewards_b.view(B * T, 1) +
                            self.reward_gamma * torch.min(tq1, tq2) *
                            (1.0 - seq_dones_b.view(B * T, 1)))

        # ── Critic update (all T steps) ───────────────────────────────────
        all_states_flat = seq_states_b.view(B * T, -1)
        all_actions_flat = next_seq_la.view(B * T, -1)

        self.critic_optimizer.zero_grad()
        q1_all = self.critic_net([all_states_flat, all_actions_flat])
        loss1 = self.criterion(q1_all, all_target_q)
        critic_loss_item = loss1.item()
        loss1.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.critic_optimizer2.zero_grad()
        q2_all = self.critic_net2([all_states_flat, all_actions_flat])
        loss2 = self.criterion(q2_all, all_target_q)
        loss2.backward()
        nn.utils.clip_grad_norm_(self.critic_net2.parameters(), max_norm=0.5)
        self.critic_optimizer2.step()

        # ── Actor update (delayed, all T steps) ──────────────────────────
        self.step_ita += 1
        if self.step_ita % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            cur_actions, _ = self.actor_net(seq_ev, seq_goal,
                                             last_action=seq_la, hidden=hidden_b,
                                             return_seq=True)  # (B, T, action_num)
            actor_loss = -self.critic_net([all_states_flat,
                                           cur_actions.view(B * T, -1)]).mean()
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
        except FileExistsError:
            pass
        torch.save(self.actor_net.state_dict(),
                   save_dir + '/' + run_name + '_dvs_actor_s' + str(episode) + '.pt')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, load_file_name):
        self.actor_net.to('cpu')
        self.actor_net.load_state_dict(torch.load(load_file_name))
        self.actor_net.to(self.device)

    def _random_minibatch(self):
        """Sample batch and return tensors on self.device."""
        minibatch = random.sample(self.memory, self.batch_size)
        B, T = self.batch_size, self.seq_len
        hidden_size = self.actor_net.gru.hidden_size

        seq_ev_arr      = np.zeros((B, T, 2, 64, 64), dtype=np.float32)
        seq_goal_arr    = np.zeros((B, T, 4),          dtype=np.float32)
        seq_nev_arr     = np.zeros((B, T, 2, 64, 64), dtype=np.float32)
        seq_ngoal_arr   = np.zeros((B, T, 4),          dtype=np.float32)
        seq_la_arr      = np.zeros((B, T, self.action_num), dtype=np.float32)
        state_arr       = np.zeros((B, self.state_num), dtype=np.float32)
        action_arr      = np.zeros((B, self.action_num), dtype=np.float32)
        reward_arr      = np.zeros((B, 1),              dtype=np.float32)
        nstate_arr      = np.zeros((B, self.state_num), dtype=np.float32)
        done_arr        = np.zeros((B, 1),              dtype=np.float32)
        hidden_arr      = np.zeros((1, B, hidden_size), dtype=np.float32)
        seq_states_arr  = np.zeros((B, T, self.state_num), dtype=np.float32)
        seq_nstates_arr = np.zeros((B, T, self.state_num), dtype=np.float32)
        seq_rewards_arr = np.zeros((B, T, 1),           dtype=np.float32)
        seq_dones_arr   = np.zeros((B, T, 1),           dtype=np.float32)

        for i, entry in enumerate(minibatch):
            (seq_ev, seq_goal, seq_nev, seq_ngoal,
             seq_la, state, action, reward, nstate, done, init_hidden,
             seq_states, seq_nstates, seq_rewards, seq_dones) = entry

            seq_ev_arr[i]         = seq_ev
            seq_goal_arr[i]       = seq_goal
            seq_nev_arr[i]        = seq_nev
            seq_ngoal_arr[i]      = seq_ngoal
            seq_la_arr[i]         = seq_la
            state_arr[i]          = state
            action_arr[i]         = action
            reward_arr[i, 0]      = reward
            nstate_arr[i]         = nstate
            done_arr[i, 0]        = float(done)
            hidden_arr[:, i, :]   = init_hidden[:, 0, :]
            seq_states_arr[i]     = seq_states
            seq_nstates_arr[i]    = seq_nstates
            seq_rewards_arr[i, :, 0] = seq_rewards
            seq_dones_arr[i, :, 0]   = seq_dones

        def t(arr):
            return torch.from_numpy(arr).to(self.device)

        return (t(seq_ev_arr), t(seq_goal_arr), t(seq_nev_arr), t(seq_ngoal_arr),
                t(seq_la_arr), t(state_arr), t(action_arr), t(reward_arr),
                t(nstate_arr), t(done_arr), t(hidden_arr),
                t(seq_states_arr), t(seq_nstates_arr), t(seq_rewards_arr), t(seq_dones_arr))

    def _hard_update(self, target, source):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.data.copy_(p.data)

    def _soft_update(self, target, source):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.target_tau) + p.data * self.target_tau)
