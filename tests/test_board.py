import pytest

from connect4.board import Board


def test_insert_some_coins():
    b = Board()
    assert b.turn() == 'O'
    b = b.insert(3)
    assert b.turn() == 'X'
    assert b == Board([0, 0, 0, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 2, 3, 4, 5, 6)
    b = b.insert(2)
    assert b.turn() == 'O'
    assert b == Board([0, 0, 0b10, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 2, 3, 4, 5, 6)
    b = b.insert(2)
    assert b.turn() == 'X'
    assert b == Board([0, 0, 0b0110, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 2, 3, 4, 5, 6)
    b = b.insert(2)
    assert b.turn() == 'O'
    assert b == Board([0, 0, 0b100110, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 2, 3, 4, 5, 6)
    b = b.insert(2)
    assert b.turn() == 'X'
    assert b == Board([0, 0, 0b01100110, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 2, 3, 4, 5, 6)
    b = b.insert(2)
    assert b.turn() == 'O'
    assert b == Board([0, 0, 0b1001100110, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 2, 3, 4, 5, 6)
    b = b.insert(2)
    assert b.turn() == 'X'
    assert b == Board([0, 0, 0b011001100110, 0b01, 0, 0, 0])
    assert b.valid_actions() == (0, 1, 3, 4, 5, 6)


def test_insert_coins_full():
    b = Board([0, 0, 0b011001100110, 0b01, 0, 0, 0])
    with pytest.raises(ValueError):
        b.insert(2)


draw_and_parse_params = [
    (
        (0, 0, 0, 0, 0, 0, 0),
        """
        | | | | | | | |
        | | | | | | | |
        | | | | | | | |
        | | | | | | | |
        | | | | | | | |
        | | | | | | | |
        """
    ),
    (
        (0, 0, 0b011001100110, 0b01, 0b10, 0b01, 0),
        """
        | | |O| | | | |
        | | |X| | | | |
        | | |O| | | | |
        | | |X| | | | |
        | | |O| | | | |
        | | |X|O|X|O| |
        """
    ),
    (
        (0b011001100110, 0b100110011001, 0b011001100110, 0b100110011001,
         0b011001100110, 0b100110011001, 0b011001100110),
        """
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        """
    ),
]


@pytest.mark.parametrize("cols,drawing", draw_and_parse_params)
def test_draw(cols, drawing):
    b = Board(cols)
    assert b.draw().strip() == "\n".join([l.strip() for l in drawing.split("\n")]).strip()


@pytest.mark.parametrize("cols,drawing", draw_and_parse_params)
def test_parse(cols, drawing):
    b = Board.parse(drawing)
    assert b.state == cols


@pytest.mark.parametrize("n_coins,drawing", [
    (
            0,
            """
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            """
    ),
    (
            9,
            """
            | | |O| | | | |
            | | |X| | | | |
            | | |O| | | | |
            | | |X| | | | |
            | | |O| | | | |
            | | |X|O|X|O| |
            """
    ),
    (
            6 * 7,
            """
            |O|X|O|X|O|X|O|
            |X|O|X|O|X|O|X|
            |O|X|O|X|O|X|O|
            |X|O|X|O|X|O|X|
            |O|X|O|X|O|X|O|
            |X|O|X|O|X|O|X|
            """
    ),
])
def test_coin_count(n_coins, drawing):
    b = Board.parse(drawing)
    assert b.number_of_coins() == n_coins


def test_equals():
    b1 = Board.parse(
        """
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        """
    )
    b2 = Board.parse(
        """
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        """
    )
    b3 = Board.parse(
        """
        |X|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        """
    )
    assert b1 == b2
    assert b2 != b3
    assert b1 != b3


def test_getitem():
    b = Board.parse(
        """
        |O|O|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        |X|X|O|X|O|X|O|
        |X|O|X|X| |O|X|
        |O|X|O|X|O|X|O|
        |X|O|X|O|X|O|X|
        """
    )
    assert b[0, 0] == 'O'
    assert b[1, 0] == 'X'
    assert b[2, 0] == 'X'
    assert b[0, 1] == 'O'
    assert b[0, 2] == 'O'
    assert b[3, 4] is None
    assert b[5, 6] == 'X'


@pytest.mark.parametrize("winner,drawing", [
    (
            None,
            """
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            """
    ),
    (
            None,
            """
            | | |O| | | | |
            | | |X| | | | |
            | | |O| | | | |
            | | |X| | | | |
            | | |O| | | | |
            | | |X|O|X|O| |
            """
    ),
    (
            '-',
            """
            |O|X|O|X|O|X|O|
            |X|O|X|X|X|O|X|
            |X|O|X|O|O|O|X|
            |X|O|X|X|X|O|X|
            |O|X|O|X|X|X|O|
            |X|O|X|O|X|O|X|
            """
    ),
    (
            None,
            """
            | | | | | | | |
            | | |X|X|X| | |
            | |O| |O|X| | |
            | |O|O|X|X|X| |
            | |O| |O| | | |
            | | | | | | | |
            """
    ),
    (
            'X',
            """
            | | | | | | | |
            | |X|X|X|X| | |
            | |O| |O|X| | |
            | |O|O|X|X|X| |
            | |O| |O| | | |
            | | | | | | | |
            """
    ),
    (
            'O',
            """
            | | | | | | | |
            | |O|X|X|X| | |
            | |O| |O|X| | |
            | |O|O|X|X|X| |
            | |O| |O| | | |
            | | | | | | | |
            """
    ),
    (
            'O',
            """
            | | | | | | | |
            | | |X|X|X| | |
            | |O| |O|X| | |
            | |O|O|X|X|X| |
            | |O| |O| | | |
            | |O| | | | | |
            """
    ),
    (
            'O',
            """
            | | | | | | |O|
            | | |X|X|X| |O|
            | |O| |O|X| |O|
            | |O|O|X|X|X|O|
            | |O| |O| | | |
            | | | | | | |X|
            """
    ),
    (
            'O',
            """
            | | | |O|O|O|O|
            | | |X|X|X| |O|
            | |O| |O|X| |X|
            | |O|O|X|X|X|O|
            | |O| |O| | | |
            | | | | | | |X|
            """
    ),
    (
            'X',
            """
            | | | |O|O|O|X|
            | | |X|X|X| |O|
            | |O| |O|X| |X|
            | |O|O|X|X|X|O|
            | |O| |O| | | |
            |X|X|X|X| | |X|
            """
    ),
    (
            'X',
            """
            |X| | | | | | |
            | |X| | | | | |
            | | |X| | | | |
            | | | |X| | | |
            | | | | | | | |
            | | | | | | | |
            """
    ),
    (
            'X',
            """
            | | | |X| | | |
            | | | | |X| | |
            | | | | | |X| |
            | | | | | | |X|
            | | | | | | | |
            | | | | | | | |
            """
    ),
    (
            'X',
            """
            | | | | | | | |
            | | | | | | | |
            | | | |X| | | |
            | | | |O|X| | |
            | | | |O|O|X| |
            | | | |O|O|O|X|
            """
    ),
    (
            'X',
            """
            | | | | | | | |
            | | | | | | | |
            |X| | | | | | |
            |O|X| | | | | |
            |O|O|X| | | | |
            |O|O|O|X| | | |
            """
    ),
    (
            'O',
            """
            | | | | | | | |
            | | | | | | | |
            |O| | | | | | |
            |X|O| | | | | |
            |X|X|O| | | | |
            |X|X|X|O| | | |
            """
    ),
    (
            'O',
            """
            | | | |O| | | |
            | | |O| | | | |
            | |O| | | | | |
            |O| | | | | | |
            | | | | | | | |
            | | | | | | | |
            """
    ),
    (
            'O',
            """
            | | | | | | |O|
            | | | | | |O| |
            | | | | |O| | |
            | | | |O| | | |
            | | | | | | | |
            | | | | | | | |
            """
    ),
    (
            'O',
            """
            | | | | | | | |
            | | | | | | | |
            | | | | | | |O|
            | | | | | |O| |
            | | | | |O| | |
            | | | |O| | | |
            """
    ),
    (
            'O',
            """
            | | | | | | | |
            | | | | | | | |
            | | | |O| | | |
            | | |O| | | | |
            | |O| | | | | |
            |O| | | | | | |
            """
    ),
    (
            'X',
            """
            | | | | | | | |
            | | | | | | | |
            | | | |X| | | |
            | | |X| | | | |
            | |X| | | | | |
            |X| | | | | | |
            """
    ),
])
def test_winner(winner, drawing):
    b = Board.parse(drawing)
    assert b.winner() == winner


def test_next_boards():
    b = Board.parse(
        """
        | | |O| | | | |
        | | |X| | | | |
        | | |O| | | | |
        | | |X| | | | |
        | | |O| | | | |
        | | |X|O|X|O| |
        """
    )
    assert b.next_boards() == [
        b.insert(0),
        b.insert(1),
        b.insert(3),
        b.insert(4),
        b.insert(5),
        b.insert(6),
    ]
