import math

import numpy as np

import gym
from gym.envs.classic_control import MountainCarEnv
from gym.wrappers.monitor import Monitor


class MountainCarEnvInherit(MountainCarEnv):

    def __init__(self):

        """
            After training/ or you do need in env write env.close- otherwise, you'll get smth like this:
                Exception ignored in: <bound method Viewer.__del__ of <gym.envs.classic_control.rendering.Viewer object at ... >>
                ImportError: sys.meta_path is None, Python is likely shutting down
        """
        super(MountainCarEnvInherit, self).__init__()


    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0: velocity = 0

        done = bool(position >= self.goal_position)
        reward = 0 if not done else 1

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}


def main():
    env = MountainCarEnvInherit() 

    for i in range(4):

        obs = env.reset()
        done = False
        while not done:

                env.render()

                action = np.rand.random(env.action_space.n)
                obs, r, done, _ = env.step(action)

                print(f"{i}: {r}, {done}, {obs}")


    env.close() 


if __name__ == "__main__":

    main()
