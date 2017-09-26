import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim

from connect4.play import generate_selfplay_session
from connect4.policy import PolicyNN

policy = PolicyNN()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def run(n_iterate=50000, n_samples=250):
    step_rewards = []

    for i in range(n_iterate):
        sessions = [generate_selfplay_session(policy) for k in range(n_samples)]

        for states, actions, reward in sessions:
            # We only have total rewards at the end of the game so far. We make the assumption that the
            # move is more relevant for the outcome, if it is at the end of the game. Therefore, we weigh
            # the rewards with 1/(moves_before_outcome)
            for j, action in enumerate(reversed(actions)):
                action.reinforce(reward / (j + 1))

            optimizer.zero_grad()
            autograd.backward(actions, [None for _ in actions])
            optimizer.step()

        rewards = [s[2] for s in sessions]
        print("iteration", i, "average_reward=", np.mean(rewards), "games won=", 100*len([s for s in sessions if s[2] > 0])/len(sessions), "%")
        step_rewards.append(np.mean(rewards))

        torch.save({
            'epoch': i,
            'state_dict': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'savepoint.bin')

    return step_rewards


run()
