import numpy as np
import matplotlib.pyplot as plt

n_arms = 2

q1 = [0.1, 0.2]
q2 = [0.9, 0.8]

p = 0.5

Q = [0, 0]

alpha = 0.1
eps = 0.1


def contextual_bandit(idx, q):
    reward = np.random.normal(q[idx], 0, size=1)

    return Q[idx] + alpha * (reward - Q[idx])


def game(step):
    reward_list = []

    for idx in range(step):
        action_idx = np.argmax(Q)
        nprand = np.random.rand(1)
        if nprand > p:
            q = q1
        else:
            q = q2

        if nprand > eps:
            Q[action_idx] = contextual_bandit(action_idx, q)
            reward_list.append(max(Q))
        else:
            action_idx = np.random.randint(2, size=1)[0]

            Q[action_idx] = contextual_bandit(action_idx, q)
            reward_list.append(max(Q))

    return reward_list


def visualize(p1):
    plt.plot(p1, label='contextual_bandit')

    plt.xlabel('steps')
    plt.ylabel('avg reward')
    plt.legend()

    plt.show()


reward = game(100000)
visualize(reward)
