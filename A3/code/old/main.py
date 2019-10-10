import argparse
import copy
import itertools
import os
import sys

import cv2
import numpy as np

import imutils
# from obj_loader import *

RECTANGLE = False   # Display bounding rectangle or not
DRAW_MATCHES = False   # Draw matches


# Load Image

# Create Matchers

# Image to grayscale

# Descriptors

# Get matches in 2 images


# Get extrinsic parameter matrix of the camera and return new homography


# Project 3D model onto pixel coordinate


# Here camera callibration procedure is defined


ref_image = load_image(sys.argv[1])
obj_path = sys.argv[2]

# Scale image

ref_grayscale = convert_to_grayscale(ref_image)
ref_described = get_descriptors(ref_grayscale)

obj = OBJ(obj_path, swapyz=True)
camera_parameters = get_camera_params()

# init video capture
cap = cv2.VideoCapture(-1)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        exit(0)

    homography, matches = get_hom(ref_described, frame)
    if RECTANGLE:
        h, w = ref_grayscale.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homography)
        frame = cv2.polylines(
            frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    if DRAW_MATCHES:
        frame = cv2.drawMatches(ref_grayscale, kp1, img2_grayscale, kp2,
                                matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if homography is not None:
        projection_matrix = get_matrix(camera_parameters, homography)
        frame = render(frame, obj, projection_matrix, ref_image, False)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
