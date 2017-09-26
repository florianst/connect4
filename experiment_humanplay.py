import sys

import click
import torch

from connect4.board import Board, BOARD_COLS
from connect4.play import select_action
from connect4.policy import PolicyNN

click.echo(click.style('Hello World!', fg='green'))

policy = PolicyNN()
saved_data = torch.load('savepoint.bin')
policy.load_state_dict(saved_data['state_dict'])
click.echo(click.style(f"Loaded a state with {saved_data['epoch']} iterations.", fg='green'))


def print_board(b):
    click.echo('\n')
    r = '|'
    for row in range(BOARD_COLS):
        r += str(row + 1)
        r += '|'
    click.echo(r)
    click.echo(b.draw())


def select_human_action(b):
    print_board(b)
    return (click.prompt('Please enter a column', type=int) - 1) % BOARD_COLS


def do_human_action(b):
    while True:
        try:
            return b.insert(select_human_action(b))
        except ValueError as e:
            click.echo(click.style(str(e), fg='red'))


b = Board()

# Decide if computer is player 1 or 2
computer_player = Board.PLAYER_1

if computer_player == Board.PLAYER_2:
    # Computer is player two, let player one play first
    b = do_human_action(b)

while True:
    # Computer moves
    a = select_action(policy, b)
    b = b.insert(a.data[0][0])

    winner = b.winner()
    if winner:
        print_board(b)
        if winner == computer_player:
            click.echo(click.style('Computer wins!', fg='green'))
        elif winner == '-':
            click.echo(click.style('Nobody wins!', fg='red'))
        else:
            print("Invalid result")
        sys.exit(0)

    # Other player moves
    b = do_human_action(b)

    winner = b.winner()
    if winner:
        print_board(b)
        if winner == '-':
            click.echo(click.style('Nobody wins!', fg='red'))
        elif winner != computer_player:
            click.echo(click.style('Human wins!', fg='red'))
        else:
            print("Invalid result")
        sys.exit(0)
