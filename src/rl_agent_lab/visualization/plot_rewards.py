import matplotlib.pyplot as plt


def moving_average(values, window=50):
    """Calculate moving average."""

    averages = []

    for i in range(len(values)):
        start = max(0, i - window + 1)

        window_values = values[start : i + 1]

        averages.append(
            sum(window_values) / len(window_values)
        )

    return averages


def plot_rewards(rewards):
    """Plot raw rewards and moving average."""

    average_rewards = moving_average(rewards)

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        alpha=0.3,
        label="Episode Reward",
    )

    plt.plot(
        average_rewards,
        label="50-Episode Average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.title("Q-Learning Training Progress")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()