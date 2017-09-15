import torch.nn as nn
import torch.nn.functional as F

from connect4.board import BOARD_COLS, BOARD_ROWS


class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(BOARD_COLS * BOARD_ROWS, 64)
        self.affine2 = nn.Linear(64, 64)
        self.affine3 = nn.Linear(64, BOARD_COLS)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores)
