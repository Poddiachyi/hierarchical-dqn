import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class Storage(object):
    def __init__(self, device):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []
        self.device = device

    def push(self, state, action, reward, next_state, done):
        mask = torch.FloatTensor([[0.0] if done else [1.0]]).to(self.device)
        action = action.unsqueeze(0).to(self.device)
        reward = torch.FloatTensor(np.array([reward])).unsqueeze(1).to(self.device)
        state = state.unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.next_states.append(next_state)

    def compute(self, ):
        self.states_tensor = torch.cat(self.states)
        self.actions_tensor = torch.cat(self.actions)
        self.rewards_tensor = torch.cat(self.rewards)
        self.masks_tensor = torch.cat(self.masks)
        self.next_states_tensor = torch.cat(self.next_states)

    def sample(self, mini_batch_size):
        batch_size = self.states_tensor.size(0)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            yield self.states_tensor[indices, :], self.actions_tensor[indices, :], self.rewards_tensor[indices, :], \
                  self.next_states_tensor[indices, :], self.masks_tensor[indices, :]

    def __len__(self):

        return len(self.states)
