import numpy as np
import random

class GridWorld:
    def __init__(self, size=10, num_walls=15):
        self.size = size
        self.num_walls = num_walls
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.agent_pos = self.start_pos
        self.grid[self.goal_pos] = 3  # Goal

        # Place walls (excluding start and goal)
        self._place_walls()
        return self._get_state()

    def _place_walls(self):
        placed = 0
        while placed < self.num_walls:
            i = np.random.randint(0, self.size)
            j = np.random.randint(0, self.size)
            if (i, j) != self.start_pos and (i, j) != self.goal_pos and self.grid[i, j] == 0:
                self.grid[i, j] = 1
                placed += 1

    def _get_state(self):
        state = np.copy(self.grid)
        i, j = self.agent_pos
        state[i, j] = 2  # Agent
        return state

    def step(self, action):
        i, j = self.agent_pos
        if action == 0 and i > 0: i -= 1  # up
        elif action == 1 and i < self.size - 1: i += 1  # down
        elif action == 2 and j > 0: j -= 1  # left
        elif action == 3 and j < self.size - 1: j += 1  # right

        if self.grid[i, j] == 1:  # Wall
            return self._get_state(), -5, False

        self.agent_pos = (i, j)

        if self.agent_pos == self.goal_pos:
            return self._get_state(), 10, True

        return self._get_state(), -1, False
