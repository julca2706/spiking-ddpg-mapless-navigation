from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import os
from sddpg_navigation.training.train_ddpg.ddpg_networks import ActorNet, CriticNet


class Agent:
    """
    Class for DDPG Agent

    Main Function:
        1. Remember: Insert new memory into the memory list

        2. Act: Generate New Action base on actor network

        3. Replay: Train networks base on mini-batch replay

        4. Save: Save actor network weights

        5. Load: Load actor network weights
    """
    def __init__(self,
                 state_num,
                 action_num,
                 rescale_state_num,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 memory_size=1000,
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 critic_lr = 1e-4,
                 actor_lr = 1e-5,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=60000,
                 epsilon_rand_decay_step=1,
                 poisson_window=50,
                 use_poisson=False,
                 policy_delay=2,
                 seq_len=10,
                 use_cuda=True):
        """

        :param state_num: number of state
        :param action_num: number of action
        :param rescale_state_num: number of rescale state
        :param actor_net_dim: dimension of actor network
        :param critic_net_dim: dimension of critic network
        :param memory_size: size of memory
        :param batch_size: size of mini-batch
        :param target_tau: update rate for target network
        :param target_update_steps: update steps for target network
        :param reward_gamma: decay of future reward
        :param actor_lr: learning rate for actor network
        :param critic_lr: learning rate for critic network
        :param epsilon_start: max value for random action
        :param epsilon_end: min value for random action
        :param epsilon_decay: steps from max to min random action
        :param epsilon_rand_decay_start: start step for epsilon start to decay
        :param epsilon_rand_decay_step: steps between epsilon decay
        :param poisson_window: window of poisson spike
        :param use_poisson: if or not use poisson spike random
        :param use_cuda: if or not use gpu
        """
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
        self.poisson_window = poisson_window
        self.use_poisson = use_poisson
        self.policy_delay = policy_delay
        self.seq_len = seq_len
        self.use_cuda = use_cuda
        '''
        Random Action
        '''
        self.epsilon = epsilon_start
        '''
        Device
        '''
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        """
        Memory
        """
        self.memory = deque(maxlen=self.memory_size)
        """
        Networks and Target Networks
        """
        self.actor_net = ActorNet(self.rescale_state_num, self.action_num,
                                  hidden1=actor_net_dim[0],
                                  hidden2=actor_net_dim[1],
                                  hidden3=actor_net_dim[2])
        self.critic_net = CriticNet(self.state_num, self.action_num,
                                    hidden1=critic_net_dim[0],
                                    hidden2=critic_net_dim[1],
                                    hidden3=critic_net_dim[2])
        self.critic_net2 = CriticNet(self.state_num, self.action_num,
                                     hidden1=critic_net_dim[0],
                                     hidden2=critic_net_dim[1],
                                     hidden3=critic_net_dim[2])
        self.target_actor_net = ActorNet(self.rescale_state_num, self.action_num,
                                         hidden1=actor_net_dim[0],
                                         hidden2=actor_net_dim[1],
                                         hidden3=actor_net_dim[2])
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
        """
        Criterion and optimizers
        """
        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.critic_lr)
        """
        Step Counter
        """
        self.step_ita = 0
        self.last_actor_loss = 0.0
        self.current_seq = []
        self.hidden_state = None

    def remember(self, state, rescale_state, action, reward, next_state, rescale_next_state, done):
        """
        Add New Memory Entry into memory deque
        :param state: current state
        :param action: current action
        :param reward: reward after action
        :param next_state: next action
        :param done: if is done
        """
        self.memory.append((state, rescale_state, action, reward, next_state, rescale_next_state, done))

    def remember_seq(self, state, rescale_state, action, reward, next_state, rescale_next_state, done):
        """
        Accumulate transitions and store sequences of length seq_len in memory.
        Pads with zeros at the front for incomplete sequences at episode end.
        """
        self.current_seq.append((state, rescale_state, action, reward, next_state, rescale_next_state, done))
        if len(self.current_seq) == self.seq_len or done:
            seq = self.current_seq
            pad_len = self.seq_len - len(seq)
            if pad_len > 0:
                zero_rstate = np.zeros_like(seq[0][1])
                padding = [(np.zeros_like(seq[0][0]), zero_rstate, np.zeros_like(seq[0][2]),
                            0, np.zeros_like(seq[0][4]), zero_rstate, False)] * pad_len
                seq = padding + seq
            seq_rescale_states = np.array([t[1] for t in seq])
            seq_rescale_next_states = np.array([t[5] for t in seq])
            last = self.current_seq[-1]
            self.memory.append((seq_rescale_states, seq_rescale_next_states,
                                last[0], last[2], last[3], last[4], last[6]))
            self.current_seq = []
        if done:
            self.hidden_state = None

    def act(self, state, explore=True, train=True):
        """
        Generate Action based on state
        :param state: current state
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: action
        """
        with torch.no_grad():
            state = np.array(state)
            if self.use_poisson:
                state = self._state_2_poisson_state(state, 1)
            state = torch.Tensor(state.reshape((1, 1, -1))).to(self.device)
            action, self.hidden_state = self.actor_net(state, self.hidden_state)
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
        return action.tolist()

    def replay(self):
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        seq_rstate_batch, seq_rnstate_batch, state_batch, action_batch, reward_batch, nstate_batch, done_batch = self._random_minibatch()
        '''
        Compuate Target Q Value
        '''
        with torch.no_grad():
            naction_batch, _ = self.target_actor_net(seq_rnstate_batch)
            noise = torch.clamp(torch.randn_like(naction_batch) * 0.1, -0.2, 0.2)
            naction_batch = torch.clamp(naction_batch + noise, 0.0, 1.0)
            next_q1 = self.target_critic_net([nstate_batch, naction_batch])
            next_q2 = self.target_critic_net2([nstate_batch, naction_batch])
            next_q = torch.min(next_q1, next_q2)
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)
        '''
        Update Critic Network
        '''
        self.critic_optimizer.zero_grad()
        current_q = self.critic_net([state_batch, action_batch])
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic_optimizer2.zero_grad()
        current_q2 = self.critic_net2([state_batch, action_batch])
        critic_loss2 = self.criterion(current_q2, target_q)
        critic_loss2.backward()
        nn.utils.clip_grad_norm_(self.critic_net2.parameters(), max_norm=1.0)
        self.critic_optimizer2.step()
        '''
        Update Actor Network and Target Networks (delayed)
        '''
        self.step_ita += 1
        if self.step_ita % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            current_action, _ = self.actor_net(seq_rstate_batch)
            actor_loss = -self.critic_net([state_batch, current_action])
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
        """
        Set Epsilon to a new value
        :param new_epsilon: new epsilon value
        :param new_decay: new epsilon decay
        """
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay

    def save(self, save_dir, episode, run_name):
        """
        Save Actor Net weights
        :param save_dir: directory for saving weights
        :param episode: number of episode
        :param run_name: name of the run
        """
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        torch.save(self.actor_net.state_dict(),
                   save_dir + '/' + run_name + '_actor_network_s' + str(episode) + '.pt')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, load_file_name):
        """
        Load Actor Net weights
        :param load_file_name: weights file name
        """
        self.actor_net.to('cpu')
        self.actor_net.load_state_dict(torch.load(load_file_name))
        self.actor_net.to(self.device)

    def _state_2_poisson_state(self, state_value, batch_size):
        """
        Transform state to spikes then transform back to state to add random
        :param state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: poisson_state
        """
        spike_state_value = state_value.reshape((batch_size, self.rescale_state_num, 1))
        state_spikes = np.random.rand(batch_size, self.rescale_state_num, self.poisson_window) < spike_state_value
        poisson_state = np.sum(state_spikes, axis=2).reshape((batch_size, -1))
        poisson_state = poisson_state / self.poisson_window
        poisson_state = poisson_state.astype(float)
        return poisson_state

    def _random_minibatch(self):
        """
        Random select mini-batch from memory
        :return: seq_rstate_batch, seq_rnstate_batch, state_batch, action_batch, reward_batch, nstate_batch, done_batch
        """
        minibatch = random.sample(self.memory, self.batch_size)
        seq_rstate_batch = np.zeros((self.batch_size, self.seq_len, self.rescale_state_num))
        seq_rnstate_batch = np.zeros((self.batch_size, self.seq_len, self.rescale_state_num))
        state_batch = np.zeros((self.batch_size, self.state_num))
        action_batch = np.zeros((self.batch_size, self.action_num))
        reward_batch = np.zeros((self.batch_size, 1))
        nstate_batch = np.zeros((self.batch_size, self.state_num))
        done_batch = np.zeros((self.batch_size, 1))
        for num in range(self.batch_size):
            seq_rstate_batch[num] = minibatch[num][0]
            seq_rnstate_batch[num] = minibatch[num][1]
            state_batch[num, :] = np.array(minibatch[num][2])
            action_batch[num, :] = np.array(minibatch[num][3])
            reward_batch[num, 0] = minibatch[num][4]
            nstate_batch[num, :] = np.array(minibatch[num][5])
            done_batch[num, 0] = minibatch[num][6]
        seq_rstate_batch = torch.Tensor(seq_rstate_batch).to(self.device)
        seq_rnstate_batch = torch.Tensor(seq_rnstate_batch).to(self.device)
        state_batch = torch.Tensor(state_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        nstate_batch = torch.Tensor(nstate_batch).to(self.device)
        done_batch = torch.Tensor(done_batch).to(self.device)
        return seq_rstate_batch, seq_rnstate_batch, state_batch, action_batch, reward_batch, nstate_batch, done_batch

    def _hard_update(self, target, source):
        """
        Hard Update Weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        """
        Soft Update weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau
                )
