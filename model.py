import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 64)
        self.value_fc1 = nn.Linear(64, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.advantage_fc1 = nn.Linear(64, 128)
        self.advantage_fc2 = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Value stream
        value = self.value_fc2(F.relu(self.value_fc1(x)))

        # Advantage stream
        advantage = self.advantage_fc2(F.relu(self.advantage_fc1(x)))

        return value + advantage - advantage.mean()

        
    
    

