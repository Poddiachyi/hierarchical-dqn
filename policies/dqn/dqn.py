import torch
import torch.nn as nn

import numpy as np

class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=32):
        super(DQN, self).__init__()

        self.num_outputs = num_outputs

        self.base = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

        self.train()

    def forward(self, x):
        return self.base(x)

    def act(self, state, epsilon=0):
        if np.random.rand() > epsilon:
            q_value = self.base(state)
            action = q_value.argmax().unsqueeze(0)
        else:
            actions = np.random.rand(self.num_outputs)
            action = torch.LongTensor([actions.argmax()])
        return action
