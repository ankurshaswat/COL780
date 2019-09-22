"""
Fill documentation for complete module
Written by ankurshaswat on 18/09/2019
"""

import argparse
import itertools
import os
import copy

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

print('Is CV3 Or Better = ', imutils.is_cv3(or_better=True))

FEATURE_EXTRACTOR = 'brisk'  # 'sift'|'surf'|'brisk'|'orb'
FEATURE_MATCHER = 'bf'  # 'bf'|'knn'
GOOD_MATCH_PERCENT = 1 # Try 0.15
LOWES_RATIO = 0.75
SCALING = 40 # Percent scale down

def get_args():
    """
    Define and parse Arguments
    """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--images", type=str, default="../InSample",
                            help="path to directory containing image group directories")
    arg_parser.add_argument("-o", "--output", type=str, default="../OutSample",
                            help="path to directory to output combined images")
    arg_parser.add_argument("-s", "--scale", type=int, default=100,
                            help="Percentage of final scaled down image relative to original")
    return arg_parser.parse_args()


def load_images(folder_path):
    """
    Load all images from a folder
    """
    image_paths = os.listdir(folder_path)
    images_loc = []
    for image_rel_path in image_paths:
        image_path = folder_path + '/' + image_rel_path
        image = cv2.imread(image_path)
        images_loc.append(image)
    return images_loc


def display_images(img1_loc, img2_loc):
    """
    Display 2 images side by side
    """
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                 constrained_layout=False, figsize=(16, 9))
    ax1.imshow(img1_loc)
    ax2.imshow(img2_loc)
    plt.show()


def display_image_with_keypoints(img1_loc, kps1_loc, img2_loc, kps2_loc):
    """
    Display 2 images and their keypoints side by side
    """
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                 figsize=(20, 8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(img1_loc, kps1_loc, None, color=(255, 0, 0)))
    ax2.imshow(cv2.drawKeypoints(img2_loc, kps2_loc, None, color=(255, 0, 0)))
    plt.show()


def display_image_with_matches(img1_loc, kps1_loc, img2_loc, kps2_loc, matches_loc):
    """
    Display 2 images and their keypoints connected by lines side by side
    """
    img = cv2.drawMatches(img1_loc, kps1_loc, img2_loc, kps2_loc, matches_loc,
                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    plt.show()


def convert_to_grayscale(image):
    """
    Convert images from RGB to Grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_descriptors(image):
    """
    Fill documentation
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


def create_matcher():
    """
    Fill documentation
    """

    cross_check = bool(FEATURE_MATCHER == 'bf')

    if FEATURE_EXTRACTOR in ('sift', 'surf'):
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    return matcher


def average(lst):
    """
    Fill documentation
    """
    sum_ = 0
    for i in lst:
        sum_ += i.distance
    return sum_ / len(lst)


def get_matches(kp1_loc, dsc1_loc, kp2_loc, dsc2_loc):
    """
    Fill documentation
    """

    if FEATURE_MATCHER == 'bf':
        raw_matches = MATCHER.match(dsc1_loc, dsc2_loc)
        raw_matches.sort(key=lambda x: x.distance)
        # print("Brute Force #matches = ", len(raw_matches))
        num_good_matches = int(len(raw_matches) * GOOD_MATCH_PERCENT)
        matches_loc = raw_matches[:num_good_matches]

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

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return (result, Ht)

# def warp_images(img1_loc, img2_loc, h_loc):
#     """
#     Fill documentation
#     """
#     rows1, cols1 = img1_loc.shape[:2]
#     rows2, cols2 = img2_loc.shape[:2]
#     print("0")

#     list_of_points_1 = np.array(
#         [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], np.float32).reshape(-1, 1, 2)
#     temp_points = np.array(
#         [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]], np.float32).reshape(-1, 1, 2)
#     print("1")

#     list_of_points_2 = cv2.perspectiveTransform(temp_points, h_loc)
#     list_of_points = np.concatenate(
#         (list_of_points_1, list_of_points_2), axis=0)
#     print(list_of_points)

#     [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
#     [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
#     print("3")

#     translation_dist = [-x_min, -y_min]
#     h_translation = np.array(
#         [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
#     print(((x_max - x_min, x_max, x_min), (y_max - y_min, y_max, y_min)))

#     output_img = cv2.warpPerspective(
#         img2_loc, h_translation.dot(h_loc), (x_max - x_min, y_max - y_min))
#     output_img[translation_dist[1]:rows1+translation_dist[1],
#                translation_dist[0]:cols1+translation_dist[0]] = img1_loc
#     print("5")

#     return output_img


ARGS = vars(get_args())

print("[INFO] loading images...")
IMAGE_GRP_FOLDERS = sorted(os.listdir(ARGS["images"]))
print('[INFO] Found following image groups -', IMAGE_GRP_FOLDERS)

MATCHER = create_matcher()

for image_grp in IMAGE_GRP_FOLDERS:
    final_image_name = image_grp+'.jpg'
    print('[INFO] working on group - '+final_image_name)
    images = load_images(ARGS['images']+'/'+image_grp)
    print("[INFO] Number of components = ", len(images))
    images = [cv2.resize(x, (((x.shape[1])*SCALING)//100, (x.shape[0]*SCALING)//100)) for x in images]
    grayscale_imgs = [convert_to_grayscale(x) for x in images]
    described_imgs = [get_descriptors(x) for x in grayscale_imgs]

    order = [i for i in range(len(images))]
    fin_hom = {}
    pair_hom = {}

    dummy = None
    for (ind1, ind2) in itertools.combinations(range(len(images)), 2):
        if ind1 >= ind2:
            continue

        img1, img1_grayscale, img1_described = images[ind1], grayscale_imgs[ind1], described_imgs[ind1]
        img2, img2_grayscale, img2_described = images[ind2], grayscale_imgs[ind2], described_imgs[ind2]

        (kp1, dsc1), (kp2, dsc2) = img1_described, img2_described
        avg_distance, points1, points2, matches = get_matches(kp1, dsc1, kp2, dsc2)
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        pair_hom[(ind1, ind2)] = (h, avg_distance, matches)

    fin_hom[0] = np.identity(pair_hom[(0,1)][0].shape[0])

    visited = [0]
    for i in range(len(images)-1):
        best_pair = (i,i)
        max_num_matches = 0
        min_distance = 10000
        for j in visited:
            for k in range(len(images)):
                if k not in visited:
                    if j > k:
                        ind11 = k
                        ind21 = j
                    else:
                        ind11 = j
                        ind21 = k

                    # print(("J&K: ", (j, k)))
                    # print(("indices: ", (ind11, ind21)))
                    if pair_hom[(ind11, ind21)][1] < min_distance or (len(pair_hom[(ind11, ind21)][2]) > max_num_matches and pair_hom[(ind11, ind21)][1] == min_distance):
                        best_pair = (ind11, ind21)
                        # print("best_pair: ", best_pair)        
                        max_num_matches = len(pair_hom[(ind11, ind21)][2])
                        min_distance = pair_hom[(ind11, ind21)][1]
        if best_pair[0] in visited:
            vis = best_pair[0]
            non_vis = best_pair[1]
        else:
            vis = best_pair[1]
            non_vis = best_pair[0]

        visited.append(non_vis)
        print(best_pair)
        print(visited)
        print(fin_hom)

        if (non_vis, vis) in pair_hom:
            fin_hom[non_vis] = np.dot(fin_hom[vis], pair_hom[(non_vis, vis)][0])
        else:
            fin_hom[non_vis] = np.dot(fin_hom[vis], np.linalg.inv(pair_hom[(vis, non_vis)][0]))

    height, width, channels = images[0].shape
    fwidth = width
    fheight = height
    base = copy.deepcopy(images[0])
    cum_trans = np.identity(pair_hom[(0,1)][0].shape[0])
    # trans[2][2] = 0
    for i in range(1, len(images)):
        # Apply panorama correction
        # fwidth += width
        # fheight += height
        # img1 = cv2.warpPerspective(images[i], fin_hom[i], (fwidth, fheight))
        base, trans = warpTwoImages(base, images[i], np.dot(cum_trans, fin_hom[i]))
        cum_trans = trans.dot(cum_trans)
        # img1[0:base.shape[0], 0:base.shape[1]] = base
        # base = copy.deepcopy(img1)

        plt.imshow(base)
        plt.show()


        # # T1 = np.matrix([[1., 0., 0. + width / 2],
        # #                   [0., 1., 0. + height / 2],
        # #                   [0., 0., 1.]])

        # # img1 = cv2.warpPerspective(img1, T1*h, (width, height))
        # # img2 = cv2.warpPerspective(img2, h, (width, height))
        # # print(img1.shape)
        # # print(img2.shape)

        # img1[0:img2.shape[0], 0:img2.shape[1]] = img2
        # plt.imshow(img1)
        # plt.show()
        # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # cnts = cv2.findContours(
        #     thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)

        # c = max(cnts, key=cv2.contourArea)

        # (x, y, w, h) = cv2.boundingRect(c)

        # img1 = img1[y:y + h, x:x + w]

        # img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img1_described = get_descriptors(img1_grayscale)
        # (kp1, dsc1) = img1_described

        # images.append(img1)
        # grayscale_imgs.append(img1_grayscale)
        # described_imgs.append(img1_described)
        # plt.imshow(img1)
        # plt.show()
