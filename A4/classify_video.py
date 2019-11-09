"""
Live classification 
"""
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
import sys
import cv2
from utils import *
from nnet import *

if __name__ == "__main__":
    VID_FEED = cv2.VideoCapture(-1)
    SIZE = None
    model = Net()
    if torch.cuda.is_available():
        print('Found Cuda')
        model = model.cuda()
    model.load("./model/test_model")

    i = 0
    TEXT = "BACKGROUND"
    with torch.no_grad():
        while True:
            RET, FRAME = VID_FEED.read()
            if not RET:
                print("Unable to capture video")
                sys.exit()
            elif SIZE is None:
                SIZE = FRAME.shape

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            FRAME = draw_game(FRAME, SIZE, TEXT)
            cv2.imshow("frame", FRAME)
            i += 1
            if i%10 == 0:
                # Get val class from nnet model
                val = 0
                if val == 0:
                    TEXT = "NEXT"
                elif val == 3:
                    TEXT = "PREV"
                elif val == 2:
                    TEXT = "PAUSE"
                else:
                    TEXT = "BACKGROUND"

                i = 0
        VID_FEED.release()
        cv2.destroyAllWindows()
