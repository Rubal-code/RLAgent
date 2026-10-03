import random


class QLearningAgent:
    """
    Q-Learning agent for a discrete environment.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        self.q_table = {}

    def get_q_values(self, state):
        """
        Get Q-values for a state.

        If the state has never been seen before,
        initialize all actions with 0.
        """

        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]

        return self.q_table[state]

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        """

        q_values = self.get_q_values(state)

        # Exploration
        if random.random() < self.exploration_rate:
            return random.randint(0, 3)

        # Exploitation
        return q_values.index(max(q_values))

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-Learning formula.
        """

        current_q = self.get_q_values(state)[action]

        if done:
            target = reward

        else:
            next_q = max(self.get_q_values(next_state))

            target = reward + self.discount_factor * next_q

        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        self.q_table[state][action] = new_q

    def decay_exploration(self):
        """
        Reduce exploration after each episode.
        """

        self.exploration_rate *= self.exploration_decay

        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate,
        )