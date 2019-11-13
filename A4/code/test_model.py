"""
Call test functions and all.
"""

import random

import numpy as np
import torch

from neural_net import NNet
from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()

    SEED = ARGS.seed

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


    NET = NNet(ARGS)

    if ARGS.load_model_path != '':
        NET.load(ARGS.load_model_path)

    NET.get_test_accuracy()
