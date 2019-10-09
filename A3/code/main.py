import argparse
import copy
import itertools
import os
import sys
import math
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from obj_loader import *

FEATURE_EXTRACTOR = 'orb'  # 'sift'|'surf'|'brisk'|'orb'
FEATURE_MATCHER = 'bf'  # 'bf'|'knn'
NUM_GOOD_MATCHES = 200
LOWES_RATIO = 0.75
SCALING = 20  # Percent scale down
RECTANGLE = False   # Display bounding rectangle or not
DRAW_MATCHES = False   # Draw matches

# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Creates matchers
def create_matcher():
    """
    Create different types of matchers to match descriptors.
    """

    cross_check = bool(FEATURE_MATCHER == 'bf')

    if FEATURE_EXTRACTOR in ('sift', 'surf'):
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    return matcher

# Image to grayscale
def convert_to_grayscale(image):
    """
    Convert images from RGB to Grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Descriptors
def get_descriptors(image):
    """
    Generate descriptors from image according to various descriptor modules
    """
    if FEATURE_EXTRACTOR == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif FEATURE_EXTRACTOR == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif FEATURE_EXTRACTOR == 'brisk':
        descriptor = cv2.BRISK_create()
    elif FEATURE_EXTRACTOR == 'orb':
        descriptor = cv2.ORB_create()

    (keypoints, features) = descriptor.detectAndCompute(image, None)
    return (keypoints, features)

# Get matches in 2 images
def get_matches(kp1_loc, dsc1_loc, kp2_loc, dsc2_loc):
    """
    Get the matching descriptors according to feature matcher.
    """

    if FEATURE_MATCHER == 'bf':
        raw_matches = MATCHER.match(dsc1_loc, dsc2_loc)
        raw_matches.sort(key=lambda x: x.distance)
        # num_good_matches = int(len(raw_matches) * GOOD_MATCH_PERCENT)
        matches_loc = raw_matches[:NUM_GOOD_MATCHES]
        # print("Brute Force #matches = ", len(raw_matches),
        #       " and avd_dist: ", average(matches_loc))

    else:
        raw_matches = MATCHER.knnMatch(dsc1_loc, dsc2_loc, 2)
        # print("KNN #matches = ", len(raw_matches))
        matches_loc = []
        for m_val, n_val in raw_matches:
            if m_val.distance < n_val.distance * LOWES_RATIO:
                matches_loc.append(m_val)

    points1_loc = np.zeros((len(matches_loc), 2), dtype=np.float32)
    points2_loc = np.zeros((len(matches_loc), 2), dtype=np.float32)

    for i, match in enumerate(matches_loc):
        points1_loc[i, :] = kp1_loc[match.queryIdx].pt
        points2_loc[i, :] = kp2_loc[match.trainIdx].pt

    return average(matches_loc), points1_loc, points2_loc, matches_loc

def average(lst):
    """
    Average x.distance from a list of x.
    """
    sum_ = 0
    for i in lst:
        sum_ += i.distance
    return sum_ / len(lst)

def display_image_with_matches(img1_loc, kps1_loc, img2_loc, kps2_loc, matches_loc):
    """
    Display 2 images and their keypoints connected by lines side by side
    """
    img = cv2.drawMatches(img1_loc, kps1_loc, img2_loc, kps2_loc, matches_loc,
                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    plt.show()

def get_hom(a, b):
    imb = convert_to_grayscale(b)
    desca = a
    descb = get_descriptors(imb)
    (kp1, dsc1), (kp2, dsc2) = desca, descb
    avg_distance, points1, points2, matches = get_matches(kp1, dsc1, kp2, dsc2)
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return h, matches

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
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w, channel = model.shape

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
def get_camera_params():
    return np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

MATCHER = create_matcher()
ref_image = load_image(sys.argv[1])
obj_path = sys.argv[2]

if SCALING != 100:
    ref_image = cv2.resize(ref_image, (((ref_image.shape[1])*SCALING)//100,
                             (ref_image.shape[0]*SCALING)//100))

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
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homography)
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    if DRAW_MATCHES:
        frame = cv2.drawMatches(ref_grayscale , kp1, img2_grayscale, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if homography is not None:
        projection_matrix = get_matrix(camera_parameters, homography)
        frame = render(frame, obj, projection_matrix, ref_image, False)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()