"""
Ping pong game
"""

import sys

import cv2
import numpy as np

import obj_loader
from utils import (init_game, draw_game, update_game,
                   load_ref_images, get_camera_params, get_homographies_contour)

if __name__ == "__main__":
    REF_IMAGES, REF_DSC = load_ref_images()
    VID_FEED = cv2.VideoCapture(-1)
    CAM_MAT = get_camera_params()
    GAME_STATE = None

    flag_g = True
    flag_b = True
    MATCH_DATA = [None, None]
    CORNERS = [None, None]
    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()
        elif GAME_STATE is None:
            SIZE = FRAME.shape
            GAME_STATE = init_game(SIZE)

        MATCH_DATA, CORNERS = get_homographies_contour(FRAME, REF_IMAGES, MATCH_DATA, CORNERS)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        FRAME = draw_game(FRAME, SIZE, GAME_STATE)
        cv2.imshow("frame", FRAME)

        PADDLE1_POS, PADDLE2_POS = None, None
        # print(CORNERS)
        if CORNERS[0] is not None:
            PADDLE1_POS = np.average(CORNERS[0], axis=0)[1]
        if CORNERS[1] is not None:
            PADDLE2_POS = np.average(CORNERS[1], axis=0)[1]

        GAME_STATE = update_game(GAME_STATE, SIZE, PADDLE1_POS, PADDLE2_POS)

    VID_FEED.release()
    cv2.destroyAllWindows()
