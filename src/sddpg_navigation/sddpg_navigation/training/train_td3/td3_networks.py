import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """ Actor Network """
    def __init__(self, state_num, action_num, hidden1=256, hidden2=256, hidden3=256, last_action_num=0):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: gru hidden dimension
        :param last_action_num: dimension of last action input (0 to disable)
        """
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_num + last_action_num, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.gru = nn.GRU(hidden2, hidden3, batch_first=True)
        self.fc3 = nn.Linear(hidden3, action_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, last_action=None, hidden=None, return_seq=False):
        if last_action is not None:
            x = torch.cat([x, last_action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x, hidden = self.gru(x, hidden)
        if return_seq:
            out = self.sigmoid(self.fc3(x))          # (B, T, action_num)
        else:
            out = self.sigmoid(self.fc3(x[:, -1, :]))  # (B, action_num)
        return out, hidden


class CriticNet(nn.Module):
    """ Critic Network"""
    def __init__(self, state_num, action_num, hidden1=512, hidden2=512, hidden3=512):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_num, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()

    def forward(self, xa):
        x, a = xa
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(torch.cat([x, a], 1)))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out
