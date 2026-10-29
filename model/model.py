# Define the model's architecture

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Model(nn.Module):
    MAX_EPOCH = 100
    LEARNING_RATE = 1e-3

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        return

    def train(self):
        return

    def eval(self):
        return

    def run(self):
        return
