# Define the model's architecture

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Model(nn.Module):
    # RGB images
    rgb_trainloader = None
    rgb_testloader = None

    # Depth maps
    depth_trainloader = None
    depth_testloader = None


    MAX_EPOCH = 100
    LEARNING_RATE = 1e-3

    def __init__(self, rgb_trainloader, rgb_testloader, depth_trainloader, depth_testloader):
        nn.Module.__init__(self)
        self.rgb_trainloader = rgb_trainloader
        self.rgb_testloader = rgb_testloader
        self.depth_trainloader = depth_trainloader
        self.depth_testloader = depth_testloader






    def forward(self, x):
        return

    def train(self):


        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.LEARNING_RATE)

        # for epoch in range(self.MAX_EPOCH):



        return

    def eval(self):
        return

    def run(self):

        return
