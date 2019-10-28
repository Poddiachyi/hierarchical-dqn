import numpy as np


class Goal(object):

    def __init__(self, low_goal=-0.6, high_goal=0.6, step=0.1):

        self.low_goal = low_goal
        self.high_goal = high_goal
        self.step = step

        self.goals = np.round(np.arange(low_goal, high_goal, step), 2)

        self.state_goal_map = {}

        self._init_map_step()

    def _init_map_step(self):

        i = 0

        for goal in self.goals:

            self.state_goal_map[goal] = i
            i += 1

    ######################## Getters ########################

    def get_goals(self):

        return self.goals

    def get_goal(self, state):

        return self.state_goal_map[np.clip(state, self.low_goal, 0.5)]

    def get_size(self):

        return self.goals.size

