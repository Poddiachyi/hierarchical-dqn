import torch.nn as nn
from .dqn.dqn import DQN


class Policy(nn.Module):
    def __init__(self, policy_name, obs_shape, action_space):
        super(Policy, self).__init__()

        self.policy = DQN(obs_shape, action_space)

    def forward(self, *args):
         raise NotImplementedError

    def act(self, x, epsilon=0):
        action = self.policy.act(x, epsilon)
        return action
        
    def get_value(self, x):
        value = self.policy(x)  # it's q value
        return value
