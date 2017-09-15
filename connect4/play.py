import numpy as np
import torch
from torch.autograd import Variable

from connect4.board import Board, BOARD_ROWS, BOARD_COLS

POSSIBLE_ACTIONS = (0, 1, 2, 3, 4, 5, 6)
REWARD_WIN = 100
REWARD_UNDECIDED = -10
REWARD_LOOSE = -100


def select_action(policy, board: Board, noise=0):
    # Get probabilities from neural network
    state = torch.from_numpy(board.matrix().reshape(BOARD_ROWS * BOARD_COLS)).float().unsqueeze(0)
    probs = policy(Variable(state))

    # Exclude any results that are not allowed
    mult_np = np.zeros(len(POSSIBLE_ACTIONS), dtype=np.float32)
    allowed_actions = board.valid_actions()
    for i in POSSIBLE_ACTIONS:
        if i in allowed_actions:
            mult_np[i] = 1
    mult = Variable(torch.from_numpy(mult_np))
    noise = Variable(torch.from_numpy(mult_np * noise))

    probs = probs * mult + noise
    if torch.sum(probs * mult).data[0] < 1e-40:
        # Neural network only offered things that are not allowed, so we go for random
        probs = probs + mult
    return probs.multinomial()


def generate_selfplay_session(policy, t_max=100):
    """
    Play game until end or for t_max rounds.
    returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.

    b = Board()

    # Decide if we are player 1 or 2
    player = np.random.choice((Board.PLAYER_1, Board.PLAYER_2), 1)

    if player == Board.PLAYER_2:
        # We are player two, let player one play first
        a = select_action(policy, b)
        b = b.insert(a.data[0][0])

    for t in range(t_max):
        # We move
        states.append(b)
        a = select_action(policy, b)
        actions.append(a)
        b = b.insert(a.data[0][0])

        winner = b.winner()
        if winner:
            if winner == player:
                total_reward = REWARD_WIN
            elif winner == '-':
                total_reward = REWARD_UNDECIDED
            else:
                print("Invalid result")
            break

        # Other player moves
        a = select_action(policy, b, noise=0.01)
        # TODO: Add more random noise to decision of other player to avoid local minima?
        b = b.insert(a.data[0][0])

        winner = b.winner()
        if winner:
            if winner == '-':
                total_reward = REWARD_UNDECIDED
            elif winner != player:
                total_reward = REWARD_LOOSE
            else:
                print("Invalid result")
            break

    return states, actions, total_reward
