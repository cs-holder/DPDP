import torch
import torch.nn as nn
import torch.nn.functional as F



class Qnet(torch.nn.Module):
    def __init__(self, dropout, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(self.dropout(x))