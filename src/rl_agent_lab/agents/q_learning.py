import random

from rl_agent_lab.agents.bellman import bellman_update


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
        current_q = self.get_q_values(state)[action]

        if done:
            next_max_q = 0.0
        else:
            next_max_q = max(self.get_q_values(next_state))

        new_q = bellman_update(
            current_q=current_q,
            reward=reward,
            next_max_q=next_max_q,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            done=done,
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