import torch.nn as nn
import torch.nn.functional as F

from connect4.board import BOARD_COLS


class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(BOARD_COLS, 128)
        self.affine2 = nn.Linear(128, 128)
        self.affine3 = nn.Linear(128, BOARD_COLS)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores)
