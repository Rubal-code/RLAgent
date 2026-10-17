
import random


class TicTacToeAgent:
    """Tabular Q-learning agent for Tic-Tac-Toe."""

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.05,
    ):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def _get_q_values(self, state, actions):
        """Return Q-values for legal actions, initializing them if needed."""
        if state not in self.q_table:
            self.q_table[state] = {}

        for action in actions:
            self.q_table[state].setdefault(action, 0.0)

        return self.q_table[state]

    def select_action(self, state, legal_actions, training=True):
        """Choose a legal action using epsilon-greedy exploration."""
        if not legal_actions:
            raise ValueError("Cannot select an action when no legal moves remain.")

        q_values = self._get_q_values(state, legal_actions)

        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        best_value = max(q_values[action] for action in legal_actions)
        best_actions = [
            action
            for action in legal_actions
            if q_values[action] == best_value
        ]
        return random.choice(best_actions)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_legal_actions,
        terminated,
    ):
        """Apply the Q-learning update rule."""
        current_q_values = self._get_q_values(state, [action])
        current_q = current_q_values[action]

        if terminated or not next_legal_actions:
            target = reward
        else:
            next_q_values = self._get_q_values(
                next_state, next_legal_actions
            )
            target = reward + self.discount_factor * max(
                next_q_values[a] for a in next_legal_actions
            )

        current_q_values[action] = current_q + self.learning_rate * (
            target - current_q
        )

    def decay_epsilon(self):
        """Reduce exploration gradually during training."""
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )
