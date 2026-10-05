class GridWorld:
    """
    Grid World environment with obstacles.

    Actions:
        0 -> UP
        1 -> DOWN
        2 -> LEFT
        3 -> RIGHT
    """

    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    def __init__(self, size=5, obstacles=None):
        self.size = size

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Default obstacles
        if obstacles is None:
            self.obstacles = {
                (0, 3),
                (1, 1),
                (1, 3),
                (2, 1),
                (3, 2),
                (3, 3),
            }
        else:
            self.obstacles = set(obstacles)

        # Start and goal can never be obstacles.
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

        self.position = self.start

    def reset(self):
        """Reset the agent to the starting position."""

        self.position = self.start

        return self.position

    def step(self, action):
        """Perform one action."""

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        row, col = self.position

        row_change, col_change = self.ACTIONS[action]

        new_row = row + row_change
        new_col = col + col_change

        new_position = (new_row, new_col)

        # Outside the grid
        if not (
            0 <= new_row < self.size
            and 0 <= new_col < self.size
        ):
            return self.position, -5, False

        # Obstacle
        if new_position in self.obstacles:
            return self.position, -5, False

        # Move
        self.position = new_position

        # Goal
        if self.position == self.goal:
            return self.position, 10, True

        # Normal movement
        return self.position, -1, False

    def render(self):
        """Print the current grid."""

        for row in range(self.size):
            line = ""

            for col in range(self.size):

                position = (row, col)

                if position == self.position:
                    line += "A "

                elif position == self.start:
                    line += "S "

                elif position == self.goal:
                    line += "G "

                elif position in self.obstacles:
                    line += "X "

                else:
                    line += ". "

            print(line)

        print()