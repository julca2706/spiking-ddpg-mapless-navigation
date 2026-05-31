import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """ Actor Network — CNN event encoder + separate state branch + GRU, no LiDAR.
        events → CNN(256) ──┐
        state[:4] → FC(64) ─┼──► GRU(322→256) → FC → action
        last_action(2) ─────┘
        CNN commented out for ablation — uncomment to enable. """
    def __init__(self, state_num, action_num, hidden3=256, last_action_num=0):
        super(ActorNet, self).__init__()

        # Event CNN: (B*T, 2, 64, 64) -> 2x conv+pool -> (B*T, 8192) -> (B, T, 256)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_events = nn.Linear(8192, 256)

        # Separate state branch: goal_dir, goal_dis, odom_lin, odom_ang -> 64
        self.fc_state = nn.Linear(state_num, 64)

        # GRU input: CNN(256) + state_feat(64) + last_action(2) = 322
        # Ablation (CNN off): state_feat(64) + last_action(2) = 66
        gru_in = 64 + last_action_num  # CNN disabled — change to 256 + 64 + last_action_num when enabled
        self.gru = nn.GRU(gru_in, hidden3, batch_first=True)
        self.fc_out = nn.Linear(hidden3, action_num)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, events, state, last_action=None, hidden=None, return_seq=False):
        # events: (B, T, 2, H, W),  state: (B, T, state_num),  last_action: (B, T, action_num)
        B, T = state.shape[:2]

        # CNN branch — disabled for ablation
        # cnn = events.view(B * T, *events.shape[2:])
        # cnn = self.elu(self.pool1(self.elu(self.conv1(cnn))))
        # cnn = self.elu(self.pool2(self.elu(self.conv2(cnn))))
        # cnn_feat = self.elu(self.fc_events(cnn.flatten(1))).view(B, T, -1)  # (B, T, 256)

        state_feat = self.relu(self.fc_state(state))  # (B, T, 64)

        parts = [state_feat]
        # parts = [cnn_feat, state_feat]  # uncomment when CNN enabled
        if last_action is not None:
            parts.append(last_action)
        x = torch.cat(parts, dim=-1)

        x, hidden = self.gru(x, hidden)

        if return_seq:
            out = self.sigmoid(self.fc_out(x))
        else:
            out = self.sigmoid(self.fc_out(x[:, -1, :]))
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
