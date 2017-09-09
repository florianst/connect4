import numpy as np
import torch.autograd as autograd
import torch.optim as optim

from connect4.play import generate_selfplay_session
from connect4.policy import PolicyNN

policy = PolicyNN()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def run(n_iterate=100, n_samples=250):
    step_rewards = []

    for i in range(n_iterate):
        sessions = [generate_selfplay_session(policy) for k in range(n_samples)]

        for states, actions, reward in sessions:
            for action in actions:
                action.reinforce(reward)

            optimizer.zero_grad()
            autograd.backward(actions, [None for _ in actions])
            optimizer.step()

        rewards = [s[2] for s in sessions]
        print("iteration", i, "average_reward=", np.mean(rewards))
        step_rewards.append(np.mean(rewards))
    return step_rewards


run()
