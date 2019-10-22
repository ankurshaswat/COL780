import sys

import cv2
import numpy as np

import obj_loader
from utils import (do_everything, compare_markers, four_point_transform, convert_to_grayscale, draw_rectangle, find_homographies, get_matrix, get_homography_from_corners,
                   load_ref_images, calculate_dist_matches, calculate_dist_corners, render, display_image_with_matched_keypoints)

if __name__ == "__main__":
    OBJ_PATH = sys.argv[1]

    OBJ = obj_loader.OBJ(OBJ_PATH, swapyz=True)
    REF_IMAGES, REF_DSC = load_ref_images()
    VID_FEED = cv2.VideoCapture(-1)

    REACHED_X, REACHED_Y = 0, 0

    # iterate = 0

    # flag_g = True
    # flag_b = True
    methods = [
    ("THRESH_BINARY", cv2.THRESH_BINARY),
    ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
    ("THRESH_TRUNC", cv2.THRESH_TRUNC),
    ("THRESH_TOZERO", cv2.THRESH_TOZERO),
    ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]

    while True:
        # box = None
        # box_g = None
        RET, FRAMED = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        SIZE = FRAMED.shape
        FOC_LEN = SIZE[1]
        CENTER = (SIZE[1]/2, SIZE[0]/2)

        CAM_MAT = np.array(
            [[FOC_LEN, 0, CENTER[0]],
             [0, FOC_LEN, CENTER[1]],
             [0, 0, 1]], dtype="double")
        
        REACHED_X, REACHED_Y = do_everything(FRAMED, REF_IMAGES, OBJ, CAM_MAT, REACHED_X, REACHED_Y)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # FRAMED = render(FRAMED, OBJ, PROJ_MAT, REF_IMAGE1, False)
        cv2.imshow("frame", FRAMED)
        # cv2.imshow("frame", thresh)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    VID_FEED.release()
    cv2.destroyAllWindows()
