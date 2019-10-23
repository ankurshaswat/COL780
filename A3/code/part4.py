"""
Move object from one visual marker to another
"""

import sys

import cv2
import numpy as np

import obj_loader
from utils import (calculate_dist_corners, get_camera_params,
                   get_matrix, load_ref_images, render, get_homographies_contour)

if __name__ == "__main__":
    OBJ_PATH = sys.argv[1]

    OBJ = obj_loader.OBJ(OBJ_PATH, swapyz=True)
    REF_IMAGES, REF_DSC = load_ref_images()
    VID_FEED = cv2.VideoCapture(-1)
    CAM_MAT = get_camera_params()

    REACHED_X, REACHED_Y = 0, 0

    MATCH_DATA = [None, None]
    CORNER_DATA = [None, None]
    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        MATCH_DATA, CORNER_DATA = get_homographies_contour(FRAME, REF_IMAGES, MATCH_DATA, CORNER_DATA)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if MATCH_DATA[0] is not None and MATCH_DATA[1] is not None:

            HOMOGRAPHY1 = MATCH_DATA[0]
            HOMOGRAPHY2 = MATCH_DATA[1]

            CORNER1 = CORNER_DATA[0]
            CORNER2 = CORNER_DATA[1]

            PROJ_MAT1, R, T = get_matrix(CAM_MAT, HOMOGRAPHY1)

            DIST = calculate_dist_corners(CORNER1, CORNER2)
            DIST_X = DIST[0]
            DIST_Y = DIST[1]

            STEP_X = DIST_X/40
            STEP_Y = DIST_Y/40

            if abs(REACHED_X) >= abs(DIST_X) or abs(REACHED_Y) >= abs(DIST_Y):
                REACHED_X = 0
                REACHED_Y = 0
            else:
                REACHED_X += STEP_X
                REACHED_Y += STEP_Y

            TRANS = np.array(
                [[1, 0, REACHED_X], [0, 1, REACHED_Y], [0, 0, 1]])
            PROJ_MAT1 = np.dot(TRANS, PROJ_MAT1)
            FRAME = render(FRAME, OBJ, PROJ_MAT1, REF_IMAGES[0], False)
            cv2.imshow("frame", FRAME)
        else:
            cv2.imshow("frame", FRAME)

    VID_FEED.release()
    cv2.destroyAllWindows()
