class GridWorld:
    """
    Simple Grid World environment.

    The agent starts at the top-left corner
    and tries to reach the goal at the bottom-right.
    """

    def __init__(self, size=5):
        self.size = size

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        self.position = self.start

    def reset(self):
        """Reset the agent to the starting position."""

        self.position = self.start

        return self.position

    def step(self, action):
        """
        Perform one action.

        Actions:
            0 -> UP
            1 -> DOWN
            2 -> LEFT
            3 -> RIGHT
        """

        row, col = self.position

        if action == 0:
            row -= 1

        elif action == 1:
            row += 1

        elif action == 2:
            col -= 1

        elif action == 3:
            col += 1

        # Prevent leaving the grid
        row = max(0, min(row, self.size - 1))
        col = max(0, min(col, self.size - 1))

        self.position = (row, col)

        # Goal reached
        if self.position == self.goal:
            reward = 10
            done = True

        else:
            reward = -1
            done = False

        return self.position, reward, done