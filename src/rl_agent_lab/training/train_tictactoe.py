import pickle
import random
from pathlib import Path

from rl_agent_lab.agents.tictactoe_agent import TicTacToeAgent
from rl_agent_lab.environments.tictactoe import TicTacToe


def train(episodes=50_000, seed=42):
    random.seed(seed)

    env = TicTacToe()
    agent = TicTacToeAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.9999,
        min_epsilon=0.05,
    )

    recent_results = []

    for episode in range(1, episodes + 1):
        env.reset()
        terminated = False
        agent_turn = True
        state = env.get_state()

        while not terminated:
            if agent_turn:
                legal_actions = env.legal_actions()
                action = agent.select_action(
                    state, legal_actions, training=True
                )
                next_state, reward, terminated, info = env.step(
                    action, player=TicTacToe.AGENT
                )

                if terminated:
                    agent.update(
                        state,
                        action,
                        reward,
                        next_state,
                        [],
                        True,
                    )
                    result = (
                        "win"
                        if info["winner"] == TicTacToe.AGENT
                        else "draw"
                        if info["winner"] is None
                        else "loss"
                    )
                    recent_results.append(result)
                else:
                    # Opponent moves before the agent's next decision.
                    opponent_action = random.choice(env.legal_actions())
                    opponent_state, opponent_reward, opponent_done, opponent_info = (
                        env.step(
                            opponent_action, player=TicTacToe.HUMAN
                        )
                    )

                    if opponent_done:
                        agent.update(
                            state,
                            action,
                            opponent_reward,
                            opponent_state,
                            [],
                            True,
                        )
                        recent_results.append(
                            "loss"
                            if opponent_info["winner"] == TicTacToe.HUMAN
                            else "draw"
                        )
                        terminated = True
                    else:
                        agent.update(
                            state,
                            action,
                            0.0,
                            opponent_state,
                            env.legal_actions(),
                            False,
                        )

                    next_state = opponent_state

            else:
                # This branch is not normally needed because the opponent
                # responds immediately after each agent move.
                next_state = state

            state = next_state
            agent_turn = True

        agent.decay_epsilon()

        if episode % 5_000 == 0:
            window = recent_results[-5_000:]
            wins = window.count("win")
            draws = window.count("draw")
            losses = window.count("loss")

            print(
                f"Episode {episode:>6}/{episodes} | "
                f"Win: {wins / len(window):.1%} | "
                f"Draw: {draws / len(window):.1%} | "
                f"Loss: {losses / len(window):.1%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "tictactoe_q_table.pkl"

    with model_path.open("wb") as file:
        pickle.dump(agent.q_table, file)

    print(f"\nTraining complete. Q-table states: {len(agent.q_table)}")
    print(f"Model saved to: {model_path}")
    return agent


if __name__ == "__main__":
    train()
