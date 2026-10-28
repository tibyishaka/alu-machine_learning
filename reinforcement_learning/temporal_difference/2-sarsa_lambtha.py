#!/usr/bin/env python3
""" SARSA(λ) """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Q: numpy.ndarray containing the q-table
    state: the current state
    epsilon: the epsilon to use for the calculation

    Returns: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ)

    env: the openAI environment instance
    Q: numpy.ndarray of shape (s,a) containing the Q table
    lambtha: the eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: the initial threshold for epsilon greedy
    min_epsilon: the minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes

    Returns: Q, the updated Q table
    """
    initial_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        E = np.zeros(Q.shape)
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            E *= gamma * lambtha
            E[state, action] += 1
            Q += alpha * delta * E
            if done:
                break
            state, action = next_state, next_action
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
    return Q
