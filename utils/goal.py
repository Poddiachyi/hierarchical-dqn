import numpy as np


class Goal(object):

    def __init__(self, min_position=-1.2, max_position=0.6, step=0.2):

        self.min_position = min_position
        self.max_position = max_position
        self.step = step

        self.goals = np.round(np.arange(self.min_position, self.max_position, step), 2)


    def get_goals(self):

        return self.goals

    def get_goal(self, goal_idx):

        return self.goals[goal_idx]

    def get_size(self):

        return self.goals.shape[0]

