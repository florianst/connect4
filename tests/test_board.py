import pytest

from connect4.board import Board


def test_insert_coins():
    b = Board()
    b = b.insert(3)
    assert b == Board([0, 0, 0, 0b01, 0, 0, 0])
    b = b.insert(2)
    assert b == Board([0, 0, 0b10, 0b01, 0, 0, 0])
    b = b.insert(2)
    assert b == Board([0, 0, 0b0110, 0b01, 0, 0, 0])
    b = b.insert(2)
    assert b == Board([0, 0, 0b100110, 0b01, 0, 0, 0])
    b = b.insert(2)
    assert b == Board([0, 0, 0b01100110, 0b01, 0, 0, 0])
    b = b.insert(2)
    assert b == Board([0, 0, 0b1001100110, 0b01, 0, 0, 0])
    b = b.insert(2)
    assert b == Board([0, 0, 0b011001100110, 0b01, 0, 0, 0])


def test_insert_coins_full():
    b = Board([0, 0, 0b011001100110, 0b01, 0, 0, 0])
    with pytest.raises(ValueError):
        b.insert(2)
