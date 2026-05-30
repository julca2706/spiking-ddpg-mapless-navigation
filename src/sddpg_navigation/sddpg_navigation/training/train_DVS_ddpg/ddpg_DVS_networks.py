import torch
import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, action_num):
        super(ActorNet, self).__init__()

        # Event branch: (B, T, 2, 64, 64) -> pool x2 -> (B*T, 8192) -> fc -> (B, T, 256)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_events = nn.Linear(8192, 256)

        # GRU processes last_action only (CNN disabled); goal concatenated after
        self.gru = nn.GRU(2, 256, batch_first=True)
        self.fc_out = nn.Linear(258, action_num)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc_events.weight)
        nn.init.zeros_(self.fc_events.bias)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, events, goal, last_action=None, hidden=None, return_seq=False):
        # events: (B, T, 2, H, W),  goal: (B, T, 2),  last_action: (B, T, action_num)
        B, T = events.shape[:2]

        # CNN branch disabled for ablation test
        # x = events.view(B * T, *events.shape[2:])
        # x = self.elu(self.pool1(self.elu(self.conv1(x))))
        # x = self.elu(self.pool2(self.elu(self.conv2(x))))
        # x = self.elu(self.fc_events(torch.flatten(x, start_dim=1)))
        # x = x.view(B, T, -1)

        if last_action is not None:
            gru_input = last_action
        else:
            gru_input = torch.zeros(B, T, 2, device=goal.device)

        output, hidden = self.gru(gru_input, hidden)
        if return_seq:
            out = self.sigmoid(self.fc_out(torch.cat([output, goal], dim=-1)))
        else:
            out = self.sigmoid(self.fc_out(torch.cat([output[:, -1, :], goal[:, -1, :]], dim=-1)))
        return out, hidden


class CriticNet(nn.Module):
    """ Critic Network — receives LiDAR state """
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
