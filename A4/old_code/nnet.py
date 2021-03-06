import torch.nn as nn
import time
import torch


class Net(nn.Module):
    def __init__(self, num_channels):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 2*num_channels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(2*num_channels, 16, 4),
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

    def save(self, classes, epc, folder_path="../models/"):
        timestamp = time.time()
        dict_save = {
            'params': self.state_dict(),
            'classes': classes
        }
        name = folder_path + str(timestamp)+'_' +str(epc)+'.model'
        torch.save(dict_save, name)
        return name

    def load(self, path):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        dict_load = torch.load(path, map_location=torch.device(device))
        self.load_state_dict(dict_load['params'])
        return dict_load['classes']
