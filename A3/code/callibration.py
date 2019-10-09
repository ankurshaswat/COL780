import cv2
import numpy as np
from cv2 import aruco

markerLength = 0.25   # Here, our measurement unit is centimetre.


def get_homography_from_corners(corners):
    pts_dst = np.array([[corners[0][0], corners[0][1]],
                        [corners[1][0], corners[1][1]],
                        [corners[2][0], corners[2][1]],
                        [corners[3][0], corners[3][1]]])
    pts_src = np.array([[0, 0],
                        [1, 0],
                        [0, 1],
                        [1, 1]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    return h, status


# def get_camera_pose(h):
#     h1 = h[:, 0]
#     h2 = h[:, 1]
#     h3 = np.cross(h1, h2)

#     val1 = np.linalg.norm(h1)
#     val2 = np.linalg.norm(h2)
#     tval = (val1+val2)/2

#     t = h[:, 2]/tval

#     return np.mat([h1, h2, h3, t])


# # init video capture
cap = cv2.VideoCapture(-1)
while True:
    ret, frame = cap.read()

    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    if not ret:
        print("Unable to capture video")
        exit(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    homography_dict = {}
    camera_pose = {}

    for ind, id_list in enumerate(ids):
        id_ = id_list[0]
        h, status = get_homography_from_corners(corners[id_])
        homography_dict[id_] = h
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength,
                                                        camera_matrix,
                                                        dist_coeffs)

        # camera_pose[id_] = get_camera_pose(h)
        camera_pose[id_] = (rvec, tvec)

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    cv2.imshow("frame", frame_markers)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cap.release()
# cv2.destroyAllWindows()
