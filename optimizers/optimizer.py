import torch
import torch.optim as optim
import torch.nn.functional as F


class Optimizer(object):
    def __init__(self,
                 meta_policy,
                 target_meta_policy,
                 policy,
                 target_policy,
                 mini_batch_size,
                 discount,
                 lr,
                 update_epochs):

        self.policy = policy
        self.target_policy = target_policy
        self.meta_policy = meta_policy
        self.target_meta_policy = target_meta_policy

        self.mini_batch_size = mini_batch_size
        self.discount = discount
        self.update_epochs = update_epochs

        self.epsilon = 1e-8
        self.gamma = 0.9

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=self.epsilon)
        self.optimizer_target = optim.Adam(self.meta_policy.parameters(), lr=lr, eps=self.epsilon)

    def update(self, meta_storage, storage):

        loss_meta = self._updata_meta_controller(meta_storage)
        loss = self._update_controller(storage)

        return loss_meta, loss

    def _updata_meta_controller(self, meta_storage):

        loss_avg = 0
        n_updates = 0

        if len(meta_storage) < self.mini_batch_size:
            return

        for e in range(self.update_epochs):

            data_generator = meta_storage.sample(self.mini_batch_size)

            for sample in data_generator:

                states, goal, rewards, next_states, masks = sample

                states = states.squeeze(1)
                goal = goal.long().squeeze(0)
                masks = masks.view(-1)

                # Compute current Q value, meta_controller takes only state and output value for every state-goal pair
                # We choose Q based on goal chosen.
                current_Q_values = self.meta_policy.get_value(states)
                current_Q_values = current_Q_values.gather(1, goal)
                # current_Q_values = self.meta_policy.get_value(states).gather(1, goal.unsqueeze(1))

                # Compute next Q value based on which goal gives max Q values
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                next_max_q = self.target_meta_policy.get_value(next_states).squeeze(1).detach().max(1)[0]
                next_Q_values = masks * next_max_q

                # Compute the target of the current Q values
                target_Q_values = rewards + (self.gamma * next_Q_values)
                # Compute Bellman error (using Huber loss)
                loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

                # Copy Q to target Q before updating parameters of Q
                self.target_meta_policy.load_state_dict(self.meta_policy.state_dict())
                # Optimize model
                self.optimizer_target.zero_grad()
                loss.backward()
                for param in self.meta_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer_target.step()
                loss_avg += loss.item()
                n_updates += 1

            return loss_avg / n_updates


    def _update_controller(self, storage):

        if len(storage) < self.mini_batch_size:
            return

        loss_avg = 0
        n_updates = 0

        for e in range(self.update_epochs):

            data_generator = storage.sample(self.mini_batch_size)

            for sample in data_generator:

                states, actions, rewards, next_states, masks = sample

                states = states.squeeze(1)
                actions = actions.long()
                masks = masks.view(-1)

                current_Q_values = self.policy.get_value(states).gather(1, actions)
                # Compute next Q value based on which goal gives max Q values
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                next_max_q = self.target_policy.get_value(next_states).squeeze(1)
                next_max_q = next_max_q.detach().max(1)[0]
                next_Q_values = masks * next_max_q
                # Compute the target of the current Q values
                target_Q_values = rewards + (self.gamma * next_Q_values)
                # Compute Bellman error (using Huber loss)
                loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

                # Copy Q to target Q before updating parameters of Q
                self.target_policy.load_state_dict(self.policy.state_dict())
                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                loss_avg += loss.item()
                n_updates += 1

            return loss_avg / n_updates

