import torch.nn as nn
import torch.nn.functional as f


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # image size will be  (224,224,3)
        self.flatten = Flatten()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(24, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*3*3, 144)
        self.fc2 = nn.Linear(144, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.pool(f.relu(self.conv3(x)))
        x = self.pool(f.relu(self.conv4(x)))
        x = self.pool(f.relu(self.conv5(x)))

        #flatten image input
        x = self.flatten(x)
        x = self.dropout(x)
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




