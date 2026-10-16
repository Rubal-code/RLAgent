
class TicTacToe:
    """A simple Tic-Tac-Toe environment for reinforcement learning."""

    HUMAN = 1
    AGENT = -1
    EMPTY = 0

    WINNING_LINES = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    def __init__(self):
        self.board = [self.EMPTY] * 9

    def reset(self):
        """Start a new game and return the board state."""
        self.board = [self.EMPTY] * 9
        return self.get_state()

    def get_state(self):
        """Return an immutable representation suitable for a Q-table."""
        return tuple(self.board)

    def legal_actions(self):
        """Return all empty cell indices."""
        return [
            index
            for index, value in enumerate(self.board)
            if value == self.EMPTY
        ]

    def winner(self):
        """Return the winning player, or None if nobody has won."""
        for a, b, c in self.WINNING_LINES:
            if (
                self.board[a] != self.EMPTY
                and self.board[a] == self.board[b] == self.board[c]
            ):
                return self.board[a]
        return None

    def is_draw(self):
        """A draw occurs when the board is full and nobody has won."""
        return self.winner() is None and self.EMPTY not in self.board

    def step(self, action, player=AGENT):
        """
        Apply a move.

        Returns:
            state, reward, terminated, info
        """
        if player not in (self.HUMAN, self.AGENT):
            raise ValueError("Player must be HUMAN (1) or AGENT (-1).")

        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        self.board[action] = player

        winning_player = self.winner()

        if winning_player is not None:
            reward = 1.0 if winning_player == self.AGENT else -1.0
            return self.get_state(), reward, True, {
                "winner": winning_player
            }

        if self.is_draw():
            return self.get_state(), 0.0, True, {"winner": None}

        return self.get_state(), 0.0, False, {"winner": None}

    def render(self):
        """Print the board."""
        symbols = {
            self.EMPTY: ".",
            self.HUMAN: "X",
            self.AGENT: "O",
        }

        for row in range(3):
            start = row * 3
            print(" ".join(symbols[self.board[i]] for i in range(start, start + 3)))
        print()
