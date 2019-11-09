from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from torch.optim import Adam
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from livelossplot import PlotLosses
import copy

IMAGE_X = 50
IMAGE_Y = 50
DROPOUT = 0.1

class Net(nn.Module):
    fin_x = 0
    fin_y = 0
    k_size = 3
    stride = 2
    pool_size = 2
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 32, self.k_size, self.stride),
            nn.Dropout2d(DROPOUT),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size, self.pool_size))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, self.k_size, self.stride),
            nn.Dropout2d(DROPOUT),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size, self.pool_size))
        
        self.drop_out = nn.Dropout(DROPOUT)

        
        tmpx = int((int((IMAGE_X-self.k_size)/self.stride)+1)/self.pool_size)
        tmpy = int((int((IMAGE_Y-self.k_size)/self.stride)+1)/self.pool_size)
        
        self.fin_x = int((int((tmpx-self.k_size)/self.stride)+1)/self.pool_size)
        self.fin_y = int((int((tmpy-self.k_size)/self.stride)+1)/self.pool_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * self.fin_x * self.fin_y, 2048),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 4),
            nn.ReLU()
        )
        
        self.sof = nn.LogSoftmax()

    def forward(self, inp):
        out = self.layer1(inp)
        out = self.layer2(out)
        out = out.view(-1, 64 * self.fin_x * self.fin_y)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sof(out)
        return out

    def save(self,path="saved_models/untitled_model"):
        torch.save(self.state_dict(), path)
    
    def load(self,path="saved_models/untitled_model"):
        self.load_state_dict(torch.load(path))
