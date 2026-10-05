import matplotlib.pyplot as plt


def plot_rewards(rewards):
    """
    Plot rewards obtained during training.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(rewards)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Progress")

    plt.grid(True)

    plt.tight_layout()

    plt.show()