import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """ Actor Network — CNN event encoder + FC + GRU, no LiDAR """
    def __init__(self, goal_num, action_num, hidden1=256, hidden2=256, hidden3=256, last_action_num=0):
        super(ActorNet, self).__init__()

        # Event CNN: (B*T, 2, 64, 64) -> 2x conv+pool -> (B*T, 8192) -> (B, T, 256)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_events = nn.Linear(8192, 256)

        self.fc1 = nn.Linear(256 + goal_num + last_action_num, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.gru = nn.GRU(hidden2, hidden3, batch_first=True)
        self.fc3 = nn.Linear(hidden3, action_num)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, events, goal, last_action=None, hidden=None, return_seq=False):
        # events: (B, T, 2, H, W),  goal: (B, T, goal_num),  last_action: (B, T, action_num)
        B, T = events.shape[:2]

        x = events.view(B * T, *events.shape[2:])
        x = self.elu(self.pool1(self.elu(self.conv1(x))))
        x = self.elu(self.pool2(self.elu(self.conv2(x))))
        x = self.elu(self.fc_events(x.flatten(1))).view(B, T, -1)  # (B, T, 256)

        if last_action is not None:
            x = torch.cat([x, goal, last_action], dim=-1)
        else:
            x = torch.cat([x, goal], dim=-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x, hidden = self.gru(x, hidden)

        if return_seq:
            out = self.sigmoid(self.fc3(x))
        else:
            out = self.sigmoid(self.fc3(x[:, -1, :]))
        return out, hidden


class CriticNet(nn.Module):
    """ Critic Network — receives full LiDAR state (unchanged from TD3) """
    def __init__(self, state_num, action_num, hidden1=512, hidden2=512, hidden3=512):
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
