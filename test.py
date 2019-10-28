import gym
import torch
import os
from policies.policy import Policy
from policies.random import RandomAgent

save_path = './saved_models'
env_name = 'MountainCar-v0'
env_num = 0
eps = 180

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(env_name)

    # policy = torch.load(os.path.join(save_path, '{}-{}.pt'.format(env_names[env_num], eps)))
    policy = RandomAgent(env.action_space, device)

    sum_reward = 0
    obs = env.reset()

    i = 0
    while True:

        with torch.no_grad():
            value, action = policy.act(obs)

        env.render()
        obs, reward, done, info = env.step(int(action.item()))
        sum_reward += reward
        if done: 
            break
        i += 1

    env.close()
    print('{} steps, reward {}'.format(i, sum_reward))

if __name__ == '__main__':
    main()
