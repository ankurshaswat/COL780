"""
Live classification
"""
import sys

import cv2
import numpy as np
import torch

from nnet import Net
from utils import TRANSFORM, draw_game

if __name__ == "__main__":
    VID_FEED = cv2.VideoCapture(-1)
    SIZE = None
    model = Net(5)
    if torch.cuda.is_available():
        print('Found Cuda')
        model = model.cuda()

    classes = model.load(sys.argv[1])

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
                img = TRANSFORM(FRAME)
                img = img[np.newaxis, :, :, :]
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                print(outputs, classes)
                # Get val class from nnet model
                TEXT = classes[predicted[0]]
                i = 0

        VID_FEED.release()
        cv2.destroyAllWindows()
