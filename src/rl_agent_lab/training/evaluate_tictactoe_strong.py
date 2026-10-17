
import pickle
import random
from pathlib import Path

from rl_agent_lab.environments.tictactoe import TicTacToe


def find_winning_move(env, player):
    """Find a move that immediately wins for the specified player."""
    for action in env.legal_actions():
        env.board[action] = player
        wins = env.winner() == player
        env.board[action] = TicTacToe.EMPTY

        if wins:
            return action

    return None


def choose_opponent_action(env):
    """Play tactically: win, block, take center, then corners."""
    winning_move = find_winning_move(env, TicTacToe.HUMAN)
    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(env, TicTacToe.AGENT)
    if blocking_move is not None:
        return blocking_move

    legal = env.legal_actions()

    if 4 in legal:
        return 4

    corners = [action for action in (0, 2, 6, 8) if action in legal]
    if corners:
        return random.choice(corners)

    return random.choice(legal)


def evaluate(episodes=1000, seed=123):
    random.seed(seed)

    model_path = Path("models") / "tictactoe_q_table.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the agent first."
        )

    with model_path.open("rb") as file:
        q_table = pickle.load(file)

    env = TicTacToe()
    results = {"win": 0, "draw": 0, "loss": 0}

    for _ in range(episodes):
        env.reset()

        while True:
            state = env.get_state()
            legal = env.legal_actions()
            values = q_table.get(state, {})

            best_value = max(values.get(a, 0.0) for a in legal)
            best_actions = [
                a for a in legal
                if values.get(a, 0.0) == best_value
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

            opponent_action = choose_opponent_action(env)
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
                break

    print(f"Strong-opponent evaluation: {episodes} games")
    for result, count in results.items():
        print(f"{result.capitalize()} rate: {count / episodes:.1%} ({count}/{episodes})")


if __name__ == "__main__":
    evaluate()
