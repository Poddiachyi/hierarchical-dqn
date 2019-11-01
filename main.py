import numpy as np
from collections import deque

import torch

import gym
from optimizers.optimizer import Optimizer

from policies.policy import Policy
from utils.storage import Storage
from utils.goal import Goal


n_eps = 200
learning_rate = 3e-3
n_steps = 500
max_grad_norm = 0.5
discount = 0.99
mini_batch_size = 256
update_epochs = 1

e_decay = 0.005
e_meta_decay = 0.02

target_policy_update = 5

seed = 42

env_name = 'MountainCar-v0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def get_intrinsic_reward(goal, state):   # should it be equal? or maybe even if agent passes position, rewards should be given

    return 1.0 if goal.item() == np.round(state[0], 2) else 0.0


def get_policies(env, goal_object):
    meta_policy = Policy(env.observation_space.shape[0], goal_object.get_size()) 
    target_meta_policy = Policy(env.observation_space.shape[0], goal_object.get_size()) 
    policy = Policy(env.observation_space.shape[0] + 1, env.action_space.n)
    target_policy = Policy(env.observation_space.shape[0] + 1, env.action_space.n)

    meta_policy.to(device)
    target_meta_policy.to(device)
    policy.to(device)
    target_policy.to(device)

    target_meta_policy.load_state_dict(meta_policy.state_dict())
    target_policy.load_state_dict(policy.state_dict())

    return meta_policy, target_meta_policy, policy, target_policy


def main():
    env = gym.make(env_name)
    goal_object = Goal(min_position=env.env.min_position, max_position=env.env.max_position, step=0.2)

    meta_policy, target_meta_policy, policy, target_policy = get_policies(env, goal_object)

    optimizer = Optimizer(meta_policy, target_meta_policy, policy, target_policy, mini_batch_size, discount, 
                          learning_rate, update_epochs)

    episode_rewards = deque(maxlen=50)

    get_meta_epsilon = lambda episode: np.exp(-episode * e_meta_decay)
    get_epsilon = lambda episode: np.exp(-episode * e_decay)

    frame = 0
    meta_frame = 0

    for eps in range(0, n_eps + 1):
        print('Episode', eps)

        if eps % 5 == 0:
            episode_rewards.append(test_env(target_meta_policy, target_policy, goal_object))
            print('Avg reward', np.mean(episode_rewards))

        storage = Storage(device=device)
        storage_meta = Storage(device=device)
        
        meta_state = env.reset()
        state = meta_state.copy()
        state = torch.FloatTensor(state).to(device)

        done = False

        for step in range(10):

            goal = meta_policy.act(meta_state, get_meta_epsilon(meta_frame))
            goal_value = torch.FloatTensor([goal_object.get_goal(int(goal.item()))]).to(device)

            print('Predicted goal', goal_value)

            goal_reached = False

            for i in range(30):

                joint_state = torch.cat((state, goal_value), axis=0).to(device)

                with torch.no_grad():
                    action = policy.act(joint_state, get_epsilon(frame))

                next_state, reward, done, _ = env.step(int(action.item()))

                intrinsic_reward = get_intrinsic_reward(goal_value, next_state)
                goal_reached = True if intrinsic_reward else False

                joint_next_state = np.concatenate([next_state, [goal_value.item]], axis=0)
                storage.push(joint_state, action, intrinsic_reward, joint_next_state, done)

                extrinsic_reward += reward

                state = next_state
                state = torch.FloatTensor(state).to(device)

                frame += 1

                if done or goal_reached:
                    break
 
            storage_meta.push(meta_state, goal, extrinsic_reward, next_state, done) # do i really pass here unnormalized goal?
            meta_state = state

            meta_frame += 1

            if done:
                break

        storage.compute()
        storage_meta.compute()

        loss_meta, loss = optimizer.update(storage_meta, storage)

        if eps % target_policy_update:
            target_meta_policy.load_state_dict(meta_policy.state_dict())
            target_policy.load_state_dict(policy.state_dict())

        with open('metrics.csv', 'a') as metrics:
            metrics.write('{},{}\n'.format(loss_meta, loss))


def test_env(meta_policy, policy, goal_object, vis=True):
    env = gym.make(env_name)
    state = env.reset()
    state = torch.FloatTensor(state).to(device)

    done = False

    extrinsic_reward = 0

    for step in range(10):

        goal = meta_policy.act(state)
        goal_value = torch.FloatTensor([goal_object.get_goal(int(goal.item()))]).to(device)

        goal_reached = False

        for i in range(30):

            joint_state = torch.cat((state, goal_value), axis=0).to(device)

            with torch.no_grad():
                action = policy.act(joint_state)

            next_state, reward, done, _ = env.step(int(action.item()))

            intrinsic_reward = get_intrinsic_reward(goal_value, next_state)
            goal_reached = True if intrinsic_reward else False

            joint_next_state = np.concatenate([next_state, [goal_value.item]], axis=0)

            extrinsic_reward += reward

            state = next_state
            state = torch.FloatTensor(state).to(device)

            if done or goal_reached:
                break

        meta_state = state

        if done:
            break

    return extrinsic_reward



if __name__ == '__main__':
    main()
