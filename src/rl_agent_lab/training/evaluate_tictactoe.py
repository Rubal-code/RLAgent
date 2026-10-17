
import pickle
import random
from pathlib import Path

from rl_agent_lab.environments.tictactoe import TicTacToe
from rl_agent_lab.training.experiment_tracker import log_experiment


def evaluate(episodes=1000, seed=123):
    random.seed(seed)

    model_path = Path("models") / "tictactoe_q_table.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run train_tictactoe first."
        )

    with model_path.open("rb") as file:
        q_table = pickle.load(file)

    env = TicTacToe()
    results = {"win": 0, "draw": 0, "loss": 0}

    for _ in range(episodes):
        env.reset()
        terminated = False

        while not terminated:
            state = env.get_state()
            legal_actions = env.legal_actions()
            state_values = q_table.get(state, {})

            # Greedy policy: select the best learned legal action.
            best_value = max(
                (state_values.get(action, 0.0) for action in legal_actions),
                default=0.0,
            )
            best_actions = [
                action
                for action in legal_actions
                if state_values.get(action, 0.0) == best_value
            ]
            action = random.choice(best_actions)

            _, _, terminated, info = env.step(
                action, player=TicTacToe.AGENT
            )

            if terminated:
                winner = info["winner"]
                results[
                    "win" if winner == TicTacToe.AGENT
                    else "draw" if winner is None
                    else "loss"
                ] += 1
                break

            # Random opponent responds.
            opponent_action = random.choice(env.legal_actions())
            _, _, terminated, info = env.step(
                opponent_action, player=TicTacToe.HUMAN
            )

            if terminated:
                winner = info["winner"]
                results[
                    "win" if winner == TicTacToe.AGENT
                    else "draw" if winner is None
                    else "loss"
                ] += 1

    print(f"Evaluation games: {episodes}")
    for result, count in results.items():
        print(
            f"{result.capitalize()} rate: "
            f"{count / episodes:.1%} ({count}/{episodes})"
        )

    metrics = {
        "win_rate": results["win"] / episodes,
        "draw_rate": results["draw"] / episodes,
        "loss_rate": results["loss"] / episodes,
    }

    log_path = log_experiment(
        algorithm="Q-Learning",
        environment="TicTacToe-vs-Random",
        seed=seed,
        episodes=episodes,
        metrics=metrics,
        model_path=str(model_path),
    )

    print(f"Experiment results logged to: {log_path}")


if __name__ == "__main__":
    evaluate()
