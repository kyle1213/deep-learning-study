import numpy as np
import matplotlib.pyplot as plt

#q에 참 action에 따른 reward가 있고, Q에 지금까지 관찰된 action에 따른 reward
#max Q에 해당하는 action을 찾아 q에 적용하고 q로부터 reward를 받고 Q갱신

n_arms = 10
alpha = 0.1
eps = 0.1
stationary_q = np.random.normal(0, 1, size=10)
eps_nonstationary_q = [0 for i in range(10)]
const_nonstationary_q = [0 for j in range(10)]

eps_Q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
greedy_Q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nonstatinary_eps_Q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nonstatinary_const_Q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

eps_sample_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
greedy_sample_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def eps_sample_average(idx):
    n = eps_sample_num[idx]
    n += 1
    reward = np.random.normal(stationary_q[idx], 1, size=1)

    return eps_Q[idx] + (1/n) * (reward - eps_Q[idx]), n


def greedy_sample_average(idx):
    n = greedy_sample_num[idx]
    n += 1
    reward = np.random.normal(stationary_q[idx], 1, size=1)

    return greedy_Q[idx] + (1/n) * (reward - greedy_Q[idx]), n


def eps_sample_average_nonstationary(idx):
    n = eps_sample_num[idx]
    n += 1
    reward = np.random.normal(nonstatinary_eps_Q[idx], 0.1, size=1)
    eps_nonstationary_q[idx] += reward

    return eps_Q[idx] + (1/n) * (reward - eps_Q[idx]), n


def constant_average(idx):
    reward = np.random.normal(nonstatinary_const_Q[idx], 0.1, size=1)
    const_nonstationary_q[idx] += reward

    return nonstatinary_const_Q[idx] + alpha * (reward - nonstatinary_const_Q[idx])


def eps_stationary(step):
    reward_list = []

    for idx in range(step):
        action_idx = np.argmax(eps_Q)
        nprand = np.random.rand(1)

        if nprand > eps:
            # greedy하게 선택
            eps_Q[action_idx], eps_sample_num[action_idx] = eps_sample_average(action_idx)
            reward_list.append(max(eps_Q))
        else:
            action_idx = np.random.randint(10, size=1)[0]

            eps_Q[action_idx], eps_sample_num[action_idx] = eps_sample_average(action_idx)
            reward_list.append(max(eps_Q))

    return reward_list


def greedy_stationary():
    reward_list = []

    for idx in range(1000):
        action_idx = np.argmax(greedy_Q)

        greedy_Q[action_idx], greedy_sample_num[action_idx] = greedy_sample_average(action_idx)
        reward_list.append(max(greedy_Q))

    return reward_list


def eps_nonstationary(step):
    reward_list = []

    for idx in range(step):
        action_idx = np.argmax(nonstatinary_eps_Q)
        nprand = np.random.rand(1)

        if nprand > eps:
            # greedy하게 선택
            nonstatinary_eps_Q[action_idx], eps_sample_num[action_idx] = eps_sample_average_nonstationary(action_idx)
            reward_list.append(max(nonstatinary_eps_Q))
        else:
            action_idx = np.random.randint(10, size=1)[0]

            nonstatinary_eps_Q[action_idx], eps_sample_num[action_idx] = eps_sample_average_nonstationary(action_idx)
            reward_list.append(max(nonstatinary_const_Q))

    return reward_list


def const_nonstationary(step):
    reward_list = []

    for idx in range(step):
        action_idx = np.argmax(nonstatinary_const_Q)
        nprand = np.random.rand(1)

        if nprand > eps:
            # greedy하게 선택
            nonstatinary_const_Q[action_idx] = constant_average(action_idx)
            reward_list.append(max(nonstatinary_const_Q))
        else:
            action_idx = np.random.randint(10, size=1)[0]

            nonstatinary_const_Q[action_idx] = constant_average(action_idx)
            reward_list.append(max(nonstatinary_const_Q))

    return reward_list


def visualize(p1, p2):
    plt.plot(p1, label='eps')
    plt.plot(p2, label='const')

    plt.xlabel('steps')
    plt.ylabel('avg reward')
    plt.legend()

    plt.show()


eps_reward = eps_nonstationary(100000)
const_reward = const_nonstationary(100000)
visualize(eps_reward, const_reward)
