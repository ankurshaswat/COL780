"""
Detect 1 visual marker and display object on it
"""

import sys

import cv2

import obj_loader
from utils import (draw_harris_kps, draw_rectangle, get_camera_params,
                   get_homographies_contour, get_matrix, load_ref_images,
                   render)

RECTANGLE = True   # Display bounding rectangle or not
DRAW_MATCHES = False   # Draw matches
DRAW_HARRIS = False

if __name__ == "__main__":

    OBJ_PATH = sys.argv[1]

    OBJ = obj_loader.OBJ(OBJ_PATH, swapyz=True)
    REF_IMAGES, REF_DSC = load_ref_images(1)
    CAM_PARAMS = get_camera_params()
    VID_FEED = cv2.VideoCapture(-1)
    MATCH_DATA = [None, None]

    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        if DRAW_HARRIS:
            FRAME = draw_harris_kps(FRAME)

        # MATCH_DATA = find_homographies(REF_DSC, FRAME)
        MATCH_DATA, _ = get_homographies_contour(FRAME, REF_IMAGES, MATCH_DATA, None)

        for ind, homography in enumerate(MATCH_DATA):
            # homography = match_tuple[0]
            if homography is not None:
                projection_matrix, R, T = get_matrix(CAM_PARAMS, homography)
                print(R, T)
                FRAME = render(FRAME, OBJ, projection_matrix,
                               REF_IMAGES[ind], False)
                if RECTANGLE:
                    FRAME, _ = draw_rectangle(
                        homography, REF_IMAGES[ind], FRAME)

        cv2.imshow('frame', FRAME)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    VID_FEED.release()
    cv2.destroyAllWindows()
