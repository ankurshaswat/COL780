"""
Move object from one visual marker to another
"""

import sys

import cv2
import numpy as np

import obj_loader
from utils import (draw_rectangle, find_homographies, get_matrix,
                   load_ref_images, calculate_dist, render)

if __name__ == "__main__":
    OBJ_PATH = sys.argv[1]

    OBJ = obj_loader.OBJ(OBJ_PATH, swapyz=True)
    REF_IMAGES, REF_DSC = load_ref_images()
    VID_FEED = cv2.VideoCapture(-1)

    REACHED_X, REACHED_Y = 0, 0

    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        SIZE = FRAME.shape
        FOC_LEN = SIZE[1]
        CENTER = (SIZE[1]/2, SIZE[0]/2)

        CAM_MAT = np.array(
            [[FOC_LEN, 0, CENTER[0]],
             [0, FOC_LEN, CENTER[1]],
             [0, 0, 1]], dtype="double")

        MATCH_DATA = find_homographies(REF_DSC, FRAME)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if MATCH_DATA[0][0] is not None and MATCH_DATA[1][0] is not None:

            HOMOGRAPHY1 = MATCH_DATA[0][0]
            HOMOGRAPHY2 = MATCH_DATA[1][0]
            MATCHES1 = MATCH_DATA[0][1]
            MATCHES2 = MATCH_DATA[1][1]
            KP1 = MATCH_DATA[0][3]
            KP2 = MATCH_DATA[1][3]

            FRAME = draw_rectangle(HOMOGRAPHY1, REF_IMAGES[0], FRAME)

            PROJ_MAT1 = get_matrix(CAM_MAT, HOMOGRAPHY1)

            DIST = calculate_dist(KP1, MATCHES1, KP2, MATCHES2)
            # print("DIST: ", DIST)
            # dist = np.average(homography_dict[4][1]-homography_dict[1][1], axis=1)[0]
            DIST_X = DIST[0]
            DIST_Y = DIST[1]

            STEP_X = DIST_X/10
            STEP_Y = DIST_Y/10

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
