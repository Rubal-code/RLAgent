def bellman_update(
    current_q,
    reward,
    next_max_q,
    learning_rate,
    discount_factor,
    done,
):
    if done:
        target = reward
    else:
        target = reward + discount_factor * next_max_q

    new_q = current_q + learning_rate * (target - current_q)

    return new_q