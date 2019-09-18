"""
Fill documentation for complete module
Written by ankurshaswat on 18/09/2019
"""

import argparse
import itertools
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

print('Is CV3 Or Better = ', imutils.is_cv3(or_better=True))

FEATURE_EXTRACTOR = 'orb'  # 'sift'|'surf'|'brisk'|'orb'
FEATURE_MATCHER = 'bf'  # 'bf'|'knn'
GOOD_MATCH_PERCENT = 1
LOWES_RATIO = 0.75


def get_args():
    """
    Define and parse Arguments
    """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--images", type=str, default="../InSample",
                            help="path to directory containing image group directories")
    arg_parser.add_argument("-o", "--output", type=str, default="../OutSample",
                            help="path to directory to output combined images")
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


def warp_images(img1_loc, img2_loc, h_loc):
    """
    Fill documentation
    """
    rows1, cols1 = img1_loc.shape[:2]
    rows2, cols2 = img2_loc.shape[:2]
    print("0")

    list_of_points_1 = np.array(
        [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], np.float32).reshape(-1, 1, 2)
    temp_points = np.array(
        [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]], np.float32).reshape(-1, 1, 2)
    print("1")

    list_of_points_2 = cv2.perspectiveTransform(temp_points, h_loc)
    list_of_points = np.concatenate(
        (list_of_points_1, list_of_points_2), axis=0)
    print(list_of_points)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    print("3")

    translation_dist = [-x_min, -y_min]
    h_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    print(((x_max - x_min, x_max, x_min), (y_max - y_min, y_max, y_min)))

    output_img = cv2.warpPerspective(
        img2_loc, h_translation.dot(h_loc), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1],
               translation_dist[0]:cols1+translation_dist[0]] = img1_loc
    print("5")

    return output_img


ARGS = vars(get_args())

print("[INFO] loading images...")
IMAGE_GRP_FOLDERS = sorted(os.listdir(ARGS["images"]))
print('Found following image groups -', IMAGE_GRP_FOLDERS)

MATCHER = create_matcher()

for image_grp in IMAGE_GRP_FOLDERS:
    final_image_name = image_grp+'.jpg'
    print('[INFO] working on image - '+final_image_name)
    images = load_images(ARGS['images']+'/'+image_grp)
    grayscale_imgs = [convert_to_grayscale(x) for x in images]
    described_imgs = [get_descriptors(x) for x in grayscale_imgs]

    pairwise_matches = []

    img1, img1_grayscale, img1_described = images[0], grayscale_imgs[0], described_imgs[0]
    (kp1, dsc1) = img1_described

    images.pop(0)
    grayscale_imgs.pop(0)
    described_imgs.pop(0)

    while len(described_imgs) > 1:
        # for (ind1, ind2) in itertools.combinations(range(len(images)), 2):
        #     img1, img1_grayscale, img1_described = images[ind1], grayscale_imgs[ind1], described_imgs[ind1]
        #     img2, img2_grayscale, img2_described = images[ind2], grayscale_imgs[ind2], described_imgs[ind2]
        min_dist = 10000
        best_match = 0
        best_match_data = (0, 0)
        for img2_ind, img2_described in enumerate(described_imgs):
            (kp2, dsc2) = img2_described
            avg_distance, points1, points2, matches = get_matches(
                kp1, dsc1, kp2, dsc2)
            print((img2_ind, avg_distance))
            if avg_distance < min_dist:
                best_match = img2_ind
                min_dist = avg_distance
                best_match_data = (points1, points2, matches)

        img2, img2_grayscale, img2_described = images[
            best_match], grayscale_imgs[best_match], described_imgs[best_match]
        (kp2, dsc2) = img2_described

        print("Final Best match: ", best_match)

        images.pop(best_match)
        grayscale_imgs.pop(best_match)
        described_imgs.pop(best_match)

        # display_image_with_keypoints(img1_grayscale, kp1, img2_grayscale, kp2)
        display_image_with_matches(
            img1_grayscale, kp1, img2_grayscale, kp2, best_match_data[2])

        h, mask = cv2.findHomography(
            best_match_data[0], best_match_data[1], cv2.RANSAC)
        height, width, channels = img2.shape

        # Apply panorama correction
        width = img1.shape[1] + img2.shape[1]
        height = img1.shape[0] + img2.shape[0]
        # print(width, height)

        # T1 = np.matrix([[1., 0., 0. + width / 2],
        #                   [0., 1., 0. + height / 2],
        #                   [0., 0., 1.]])

        # img1 = cv2.warpPerspective(img1, T1*h, (width, height))
        img1 = cv2.warpPerspective(img1, h, (width, height))
        # img2 = cv2.warpPerspective(img2, h, (width, height))
        # print(img1.shape)
        # print(img2.shape)
        plt.imshow(img1)
        plt.show()

        img1[0:img2.shape[0], 0:img2.shape[1]] = img2
        # img1 = warp_images(img2, img1, h)
        plt.imshow(img1)
        plt.show()
#         # print()
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)

        (x, y, w, h) = cv2.boundingRect(c)

        img1 = img1[y:y + h, x:x + w]

        img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1_described = get_descriptors(img1_grayscale)
        (kp1, dsc1) = img1_described

#         print(img1.shape)
#         # scale_percent = 50
#         width = int(img2.shape[1])
#         height = int(img2.shape[0])
#         dim = (width, height)
#         # resize image
#         img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
#         print(img1.shape)

        plt.imshow(img1)
        plt.show()

#     # for img1_ind in range(len(described_imgs)):
#     # 	for img2_ind in range(len(described_imgs)):
#     # 		if(img1_ind <= img2_ind):
#     # 			continue
#     # 		else:
#     # 			(keypoints1, descriptors1) = described_imgs[img1_ind]
#     # 			(keypoints2, descriptors2) = described_imgs[img2_ind]

#     # 			min_distance, points1, points2 = get_matches(
#     # 				keypoints1, descriptors1, keypoints2, descriptors2)

#     # 			h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

#     # 			pairwise_matches.append(
#     # 				(img1_ind, img2_ind, min_distance, h, mask))

# # References
# # https://colab.research.google.com/drive/11Md7HWh2ZV6_g3iCYSUw76VNr4HzxcX5#scrollTo=6eHgWAorE9gf
# # https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

# #    # Use homography
# #   height, width, channels = im2.shape
# #   im1Reg = cv2.warpPerspective(im1, h, (width, height))
