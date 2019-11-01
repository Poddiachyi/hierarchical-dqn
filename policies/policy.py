import torch.nn as nn
from .dqn.dqn import DQN


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()

        self.policy = DQN(obs_shape, action_space)

    def forward(self, *args):
         raise NotImplementedError

    def act(self, x, epsilon=0):
        action = self.policy.act(x, epsilon)
        return action
        
    def get_value(self, x):
        q_value = self.policy(x)
        return q_value
