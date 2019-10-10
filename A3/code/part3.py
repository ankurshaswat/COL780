"""
Detect 1 visual marker and display object on it
"""

import sys

import cv2
import numpy as np

import obj_loader
from utils import (convert_to_grayscale, draw_harris_kps, get_camera_params,
                   get_descriptors, get_hom, get_matrix, load_image, render)

RECTANGLE = True   # Display bounding rectangle or not
DRAW_MATCHES = False   # Draw matches
MARKERS_PATHS = ['../markers/0.png', '../markers/1.png', '../markers/2.png']

if __name__ == "__main__":

    OBJ_PATH = sys.argv[1]

    REF_IMAGES = []

    for marker_path in MARKERS_PATHS:
        REF_IMAGES.append(load_image(marker_path, False))

    REF_DSC = []

    for img in REF_IMAGES:
        img_grayscale = convert_to_grayscale(img)
        REF_DSC.append(get_descriptors(img_grayscale))

    OBJ = obj_loader.OBJ(OBJ_PATH, swapyz=True)
    CAM_PARAMS = get_camera_params()

    # Init video capture
    VID_FEED = cv2.VideoCapture(-1)

    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        FRAME = draw_harris_kps(FRAME)

        MATCH_LIST = []  # (homography,matches,avg_dist)

        for descriptor in REF_DSC:
            MATCH_LIST.append(get_hom(descriptor, FRAME))

        for match_tuple in MATCH_LIST:
            homography = match_tuple[0]
            if homography is not None:
                # projection_matrix = get_matrix(camera_parameters, homography)
                # frame = render(frame, obj, projection_matrix, ref_images[0], False)

                if RECTANGLE:
                    h, w, _ = REF_IMAGES[0].shape
                    pts = np.array([[0, 0], [0, h - 1], [w - 1, h - 1],
                                    [w - 1, 0]], dtype='float32').reshape((-1, 1, 2))
                    dst = cv2.perspectiveTransform(pts, homography)
                    FRAME = cv2.polylines(
                        FRAME, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        cv2.imshow('frame', FRAME)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    VID_FEED.release()
    cv2.destroyAllWindows()
