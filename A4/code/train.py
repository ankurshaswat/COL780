"""
Call train functions and all.
"""

import random

import numpy as np
import torch

from neural_net import NNet
from utils import parse_args

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

if __name__ == "__main__":
    ARGS = parse_args()

    NET = NNet(ARGS)

    NET.train()
