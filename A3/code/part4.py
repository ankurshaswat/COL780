"""
Move object from one visual marker to another
"""

import sys

import cv2
import numpy as np
from cv2 import aruco

from utils import get_homography_from_corners, get_matrix, render

if __name__ == "__main__":
    # Init video capture
    VID_FEED = cv2.VideoCapture(-1)

    REACH_X_FLAG = 0
    REACH_Y_FLAG = 0

    while True:
        RET, FRAME = VID_FEED.read()

        SIZE = FRAME.shape
        FOCAL_LENGTH = SIZE[1]
        CENTER = (SIZE[1]/2, SIZE[0]/2)
        CAM_MATRIX = np.array(
            [[FOCAL_LENGTH, 0, CENTER[0]],
             [0, FOCAL_LENGTH, CENTER[1]],
             [0, 0, 1]], dtype="double"
        )

        if not RET:
            print("Unable to capture video")
            sys.exit()

        gray = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        homography_dict = {}
        camera_pose = {}
        if ids is not None:
            for ind, id_list in enumerate(ids):
                id_ = id_list[0]
                homography, status = get_homography_from_corners(corners[ind])
                if homography is not None:
                    # projection_matrix = get_matrix(camera_matrix, homography)
                    # frame = render(frame, obj, projection_matrix, ref_image, False)
                    homography_dict[id_] = (homography, corners[ind])
                # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength,
                    # camera_matrix,
                    # dist_coeffs)
                # print(rvec)
                # print(tvec)
                # camera_pose[id_] = get_camera_pose(h)
                # camera_pose[id_] = (rvec, tvec)
        print(homography_dict.keys())
        if 1 in homography_dict and 4 in homography_dict:
            pts = ref_image
            dst = cv2.perspectiveTransform(
                np.array([pts]), homography_dict[1][0])
            frame_markers = cv2.polylines(
                FRAME, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            projection_matrix = get_matrix(
                CAM_MATRIX, homography_dict[1][0])
            if 4 in homography_dict:
                # dst2 = cv2.perspectiveTransform(np.array([pts]),homography_dict[2][0])
                # frame_markers = cv2.polylines(frame_markers,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                dist = np.average(
                    homography_dict[4][1]-homography_dict[1][1], axis=1)[0]
                distx = dist[0]
                disty = dist[1]
                print("dst: ", dist, " ", (REACH_X_FLAG, REACH_Y_FLAG),
                      " ", homography_dict[1][1].shape)
                stepx = distx/100
                stepy = disty/100
                print("step: ", (stepx, stepy))
                if abs(REACH_X_FLAG) >= abs(distx) or abs(REACH_Y_FLAG) >= abs(disty):
                    REACH_X_FLAG = 0
                    REACH_Y_FLAG = 0
                else:
                    REACH_X_FLAG += stepx
                    REACH_Y_FLAG += stepy
                    # print("projection_matrix: ", projection_matrix)
                    # projection_matrix = np.dot(translate, projection_matrix)
                translate = np.array(
                    [[1, 0, REACH_X_FLAG], [0, 1, REACH_Y_FLAG], [0, 0, 1]])
                projection_matrix = np.dot(translate, projection_matrix)
                # projection_matrix = get_matrix(camera_matrix, homography_dict[1][0], translate)
            # else:
                # projection_matrix = get_matrix(camera_matrix, homography_dict[1][0])
            frame_markers = render(
                frame_markers, obj, projection_matrix, ref_image, False)
            cv2.imshow("frame", frame_markers)
        else:
            cv2.imshow("frame", FRAME)

        # frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
