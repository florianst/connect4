import click
import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.optim as optim

from connect4.play import generate_session, make_opponent_random, make_opponent_random_with_winning_move, make_opponent_selfplay
from connect4.policy import PolicyNN

policy = PolicyNN()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


@click.command()
@click.option('--iterations', default=50000)
@click.option('--cuda/--no-cuda', default=False)
@click.option('--samples', default=250)
@click.option('--reward-strategy', default='last_move', type=click.Choice(['last_move', 'all_moves']))
@click.option('--path', type=click.Path(), default='savepoint.bin')
@click.option('--opponent', default='self', type=click.Choice(['self', 'random', 'random_with_winning_move']))
@click.option('--selfplay-noise', default=0.01, type=click.FLOAT)
def run(iterations, samples, path, reward_strategy, opponent, selfplay_noise, cuda):
    step_rewards = []

    if cuda:
        policy.cuda()

    if opponent == 'self':
        opponent = make_opponent_selfplay(noise=selfplay_noise, cuda=cuda)
    elif opponent == 'random':
        opponent = make_opponent_random()
    elif opponent == 'random_with_winning_move':
        opponent = make_opponent_random_with_winning_move()

    for i in range(iterations):
        t = time.time()
        sessions = [generate_session(policy, opponent, cuda) for k in range(samples)]

        for states, actions, reward in sessions:
            # We only have total rewards at the end of the game so far. We make the assumption that the
            # move is more relevant for the outcome, if it is at the end of the game. Therefore, we weigh
            # the rewards with 1/(moves_before_outcome)
            for j, action in enumerate(reversed(actions)):
                if j == 0 or reward_strategy == 'all_moves':
                    action.reinforce(reward)
                else:
                    action.reinforce(0)

            optimizer.zero_grad()
            autograd.backward(actions, [None for _ in actions])
            optimizer.step()

        rewards = [s[2] for s in sessions]
        print("iteration", i, "average_reward=", np.mean(rewards),
              "games won=", 100*len([s for s in sessions if s[2] > 0])/len(sessions), "%",
              "took=", time.time() - t, "s")
        step_rewards.append(np.mean(rewards))

        if i % 10 == 0:
            torch.save({
                'epoch': i,
                'state_dict': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)

    return step_rewards


if __name__ == '__main__':
    run()
