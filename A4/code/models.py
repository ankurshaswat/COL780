"""
File to describe model of neural net
"""
import torch.nn as nn


class Net(nn.Module):
    """
    Neural net model
    """

    def __init__(self, num_channels, args):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 2*num_channels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(2*num_channels, 16, 4),
            nn.Dropout2d(args.dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(
            nn.Linear(16*10*10, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 4),
            nn.ReLU(),
        )

    def forward(self, inp):
        out = self.layer1(inp)
        out = self.layer2(out)
        out = out.view(-1, 1600)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
