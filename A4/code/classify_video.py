"""
Live classification
"""
import sys

import cv2
import numpy as np
import torch

from neural_net import NNet
from utils import TRANSFORM, draw_game, parse_args

if __name__ == "__main__":
    ARGS = parse_args()

    NET = NNet(ARGS)

    if ARGS.load_model_path != '':
        NET.load(ARGS.load_model_path)

    VID_FEED = cv2.VideoCapture(-1)
    SIZE = None

    i = 0
    TEXT = ""
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

            if i % 4 == 0:
                IMG = TRANSFORM(FRAME)
                IMG = IMG[np.newaxis, :, :, :]
                TEXT = NET.predict(IMG)
                i = 0

        VID_FEED.release()
        cv2.destroyAllWindows()
