import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Network(nn.Module):
    def __init__(self, input_size=3, action=1):
        super(Network, self).__init__()
        self.input_size = input_size  # to define input neurons, currently only 1 image (3 channels) but useful if sensors are used
        self.action = action  # 3 possible actions, turning left, right or straight (some degrees)
        # image size will be  (224,224,3) (if just using images, input size will be 3)
        self.flatten = Flatten()
        self.conv1 = nn.Conv2d(input_size, 24, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(24, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 144)
        self.fc2 = nn.Linear(144, action)  # need to read about making the output 3 (action should be set to 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, current):
        current = self.pool(f.relu(self.conv1(current)))
        current = self.pool(f.relu(self.conv2(current)))
        current = self.pool(f.relu(self.conv3(current)))
        current = self.pool(f.relu(self.conv4(current)))
        current = self.pool(f.relu(self.conv5(current)))

        # flatten image input
        current = self.flatten(current)
        current = self.dropout(current)
        current = f.relu(self.fc1(current))
        current = self.dropout(current)
        res = self.fc2(current)
        return res


