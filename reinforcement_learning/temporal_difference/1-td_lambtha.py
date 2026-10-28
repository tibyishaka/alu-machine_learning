#!/usr/bin/env python3
""" TD(λ) """

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm

    env: the openAI environment instance
    V: numpy.ndarray of shape (s,) containing the value estimate
    policy: function that takes in a state and returns the next action
    lambtha: the eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate

    Returns: V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        E = np.zeros(V.shape)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            delta = reward + gamma * V[next_state] - V[state]
            E *= gamma * lambtha
            E[state] += 1
            V += alpha * delta * E
            if done:
                break
            state = next_state
    return V
