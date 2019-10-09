import cv2
import numpy as np
from cv2 import aruco
from obj_loader import *
import sys
import math

markerLength = 0.25   # Here, our measurement unit is centimetre.
obj_path = sys.argv[1]

obj = OBJ(obj_path, swapyz=True)
ref_image = np.array([[0, 0],
                        [1000, 0],
                        [0, 1000],
                        [1000, 1000]],dtype=np.float32)

def get_homography_from_corners(corners):
    global ref_image
    # print("internal: ", corners)
    pts_dst = np.array([[corners[0][0][0], corners[0][0][1]],
                        [corners[0][1][0], corners[0][1][1]],
                        [corners[0][2][0], corners[0][2][1]],
                        [corners[0][3][0], corners[0][3][1]]])
    pts_src = ref_image

    h, status = cv2.findHomography(pts_src, pts_dst)
    return h, status

# Get extrinsic parameter matrix of the camera and return new homography 
def get_matrix(cp, h):
    Rt = np.dot(np.linalg.inv(cp),h)
    r1 = Rt[:, 0]
    r2 = Rt[:, 1]
    t = Rt[:, 2]
    norm = math.sqrt(np.linalg.norm(r1, 2) * np.linalg.norm(r2, 2))
    r1 = r1/norm
    r2 = r2/norm
    t = t/norm
    c = r1+r2
    p = np.cross(r1, r2)
    d = np.cross(c,p)
    r1_ = (1/math.sqrt(2))*(c/np.linalg.norm(c,2) + d/np.linalg.norm(d,2))
    r2_ = (1/math.sqrt(2))*(c/np.linalg.norm(c,2) - d/np.linalg.norm(d,2))
    r3_ = np.cross(r1_, r2_)
    proj = np.stack((r1_, r2_, r3_, t)).T
    return np.dot(cp, proj)

# Project 3D model onto pixel coordinate
def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

# Here camera callibration procedure is defined
# def get_camera_params():
#     return np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

# def get_camera_pose(h):
#     h1 = h[:, 0]
#     h2 = h[:, 1]
#     h3 = np.cross(h1, h2)

#     val1 = np.linalg.norm(h1)
#     val2 = np.linalg.norm(h2)
#     tval = (val1+val2)/2

#     t = h[:, 2]/tval

#     return np.mat([h1, h2, h3, t])


# camera_parameters = get_camera_params()
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
    if ids is not None:
        for ind, id_list in enumerate(ids):
            id_ = id_list[0]
            homography, status = get_homography_from_corners(corners[ind])
            if homography is not None:
                # projection_matrix = get_matrix(camera_matrix, homography)
                # frame = render(frame, obj, projection_matrix, ref_image, False)
                homography_dict[id_] = homography
            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength,
                                                            # camera_matrix,
                                                            # dist_coeffs)
            # print(rvec)
            # print(tvec)
            # camera_pose[id_] = get_camera_pose(h)
            # camera_pose[id_] = (rvec, tvec)
    print(homography_dict.keys())
    if 1 in homography_dict:
        pts = ref_image
        dst = cv2.perspectiveTransform(np.array([pts]),homography_dict[1])
        frame_markers = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        projection_matrix = get_matrix(camera_matrix, homography)
        frame_markers = render(frame_markers, obj, projection_matrix, ref_image, False)
        cv2.imshow("frame", frame_markers)
    else:
        cv2.imshow("frame", frame)

    # frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cap.release()
# cv2.destroyAllWindows()
