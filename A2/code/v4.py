"""
Python script to generate panaroma from multiple images
Written by ankurshaswat on 18/09/2019
"""

import argparse
import copy
import itertools
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

print('Is CV3 Or Better = ', imutils.is_cv3(or_better=True))

FEATURE_EXTRACTOR = 'brisk'  # 'sift'|'surf'|'brisk'|'orb'
FEATURE_MATCHER = 'bf'  # 'bf'|'knn'
NUM_GOOD_MATCHES = 200
LOWES_RATIO = 0.75
SCALING = 100  # Percent scale down

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())


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
    args = vars(arg_parser.parse_args())

    print('[ARGUMENTS LIST]')
    for arg in args:
        print('[ARG]{}:{}'.format(arg, args[arg]))
    return args


def load_images(folder_path):
    """
    Load all images from a folder
    """
    image_paths = sorted(os.listdir(folder_path))
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


def average(lst):
    """
    Average x.distance from a list of x.
    """
    sum_ = 0
    for i in lst:
        sum_ += i.distance
    return sum_ / len(lst)


def get_matches(kp1_loc, dsc1_loc, kp2_loc, dsc2_loc):
    """
    Get the matching descriptors according to feature matcher.
    """

    if FEATURE_MATCHER == 'bf':
        raw_matches = MATCHER.match(dsc1_loc, dsc2_loc)
        raw_matches.sort(key=lambda x: x.distance)
        # num_good_matches = int(len(raw_matches) * GOOD_MATCH_PERCENT)
        matches_loc = raw_matches[:NUM_GOOD_MATCHES]
        print("Brute Force #matches = ", len(raw_matches),
              " and avd_dist: ", average(matches_loc))

    else:
        raw_matches = MATCHER.knnMatch(dsc1_loc, dsc2_loc, 2)
        print("KNN #matches = ", len(raw_matches))
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


def simple_combine(img1_loc, img2_loc, h_val):
    width = img1_loc.shape[1] + img2_loc.shape[1]
    height = img1_loc.shape[0] + img2_loc.shape[0]
    img2_loc = cv2.warpPerspective(img2_loc, h_val, (width, height))
    img2_loc[0:img1_loc.shape[0], 0:img1_loc.shape[1]] = img1_loc
    gray = cv2.cvtColor(img2_loc, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)

    img = img2_loc[y:y + h, x:x + w]
    return img


def combine_images(img1_loc, img2_loc, h_val):
    """
    Warp img2 to img1 with homograph h_val
    """

    height1, width1 = img1_loc.shape[:2]
    height2, width2 = img2_loc.shape[:2]
    pts1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]],
                    np.float32).reshape(-1, 1, 2)
    pts2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]],
                    np.float32).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, h_val)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel())
    t_val = [-xmin, -ymin]
    h_translation = np.array(
        [[1, 0, t_val[0]], [0, 1, t_val[1]], [0, 0, 1]])  # translate

    warped_image = cv2.warpPerspective(
        img2_loc, h_translation.dot(h_val), (xmax-xmin, ymax-ymin))

    # plt.imshow(warped_image)
    # plt.show()


    # result1[:, :] = warped_image
    # img1_loc = np.array(img1_loc)
    # img1 = np.sum(img1_loc, axis=2)
    # indices1 = np.argwhere((img1 != 0))
    # indices = indices1 + [t_val[1], t_val[0]]
    # result1[tuple(zip(*indices))] = img1_loc[tuple(zip(*indices1))]

    # plt.imshow(result1)
    # plt.show()

    # result2[t_val[1]:t_val[1]+height1, t_val[0]:t_val[0]+width1] = img1_loc
    # warped_image_arr = np.array(warped_image)
    # warped_img_sum = np.sum(warped_image_arr, axis=2)
    # indices1 = np.argwhere((warped_img_sum != 0))
    # result2[tuple(zip(*indices1))] = warped_image[tuple(zip(*indices1))]

    # plt.imshow(result2)
    # plt.show()

    result1 = np.zeros((ymax-ymin, xmax-xmin, 3), np.uint8)
    result1[:, :] = warped_image
    img1_arr = np.array(result1)
    img1 = np.sum(img1_arr, axis=2)
    # print(img1)
    mask1 = 255*(img1 > 0)
    # print(mask1)
    # plt.imshow(mask1)
    # plt.show()

    result2 = np.zeros((ymax-ymin, xmax-xmin, 3), np.uint8)
    result2[t_val[1]:t_val[1]+height1, t_val[0]:t_val[0]+width1] = img1_loc
    img2_loc = np.array(result2)
    img2 = np.sum(img2_loc, axis=2)
    mask2 = 255*(img2 > 0)
    # plt.imshow(mask2)
    # plt.show()

    blender = cv2.detail_MultiBandBlender()
    blender.prepare((0,0,mask1.shape[0],mask1.shape[1]))

    img2 = np.int16(img2)
    img1 = np.int16(img1)
    mask2 = np.int16(mask2)
    mask1 = np.int16(mask1)

    plt.imshow(img2)
    plt.show()
    plt.imshow(mask2)
    plt.show()

    blender.feed(img2,mask2,(0,0))
    blender.feed(img1,mask1,(0,0))
    result_img,result_mask = blender.blend()

    plt.imshow(result_img)
    plt.show()
    # blank_image = np.zeros((ymax-ymin, xmax-xmin, 3), np.uint8)
    # mask = np.zeros((ymax-ymin, xmax-xmin, 3), np.uint8)
    # mask[t_val[1]:t_val[1]+height1, t_val[0]:t_val[0]+width1] = 255
    # center = (t_val[1]+height1//2, t_val[0]+width1//2)
    # blank_image[t_val[1]:t_val[1]+height1, t_val[0]:t_val[0]+width1] = img1_loc

    # mixed_clone = cv2.seamlessClone(
    #     blank_image, result, mask, center, cv2.MIXED_CLONE)

    # plt.imshow(blank_image)
    # plt.show()

    alpha = 0.5
    dst = cv2.addWeighted(result1, alpha, result2, 1-alpha, 0.0)

    plt.imshow(dst)
    plt.show()
    result = dst
    # for x_val in range(0, height1):
    #     for y_val in range(0, width1):
    #         if not (img1_loc[x_val, y_val] == [0, 0, 0]).all():
    #             result[x_val+t_val[1], y_val+t_val[0]] = img1_loc[x_val, y_val]
    #             # print(img1_loc[x_val, y_val].shape)
    #         # else:
    #             # print(img1_loc[x_val, y_val])
    # result[t_val[1]:height1+t_val[1], t_val[0]:width1+t_val[0]] = img1_loc
    return (result, h_translation)


def correct_hom(img2_loc, h_val):
    # height1, width1 = img1_loc.shape[:2]
    height2, width2 = img2_loc.shape[:2]
    # pts1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]],
    # np.float32).reshape(-1, 1, 2)
    pts2 = np.array([[0, 0], [0, height2], [width2, height2], [
                    width2, 0]], np.float32).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, h_val)
    # pts = np.concatenate((pts2), axis=0)
    # pts2_ = cv2.warpPerspective(img2_loc, h_translation.dot(h_val), (xmax-xmin, ymax-ymin))
    pts = pts2_
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    # if xmin <= 0 or ymin <= 0:
    print("Left correction homography")
    t_val = [-xmin, -ymin]
    h_translation = np.array(
        [[1, 0, t_val[0]], [0, 1, t_val[1]], [0, 0, 1]])  # translate
    # else:
    # print("Right correction homography: ", (xmax, width2), (ymax, height2))
    # t_val = [-1*(xmax-width2), -1*(ymax-height2)]
    # h_translation = np.array(
    #     [[1, 0, t_val[0]], [0, 1, t_val[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(
        img2_loc, h_translation.dot(h_val), (xmax-xmin, ymax-ymin))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    result = result[y:y + h, x:x + w]

    return result


def get_relative_position(img_1, points_1, img_2, points_2):
    """
    Get relative positioning of two images.
    """
    height1, width1 = img_1.shape[:2]
    avg1 = np.mean(points_1, axis=0)

    height2, width2 = img_2.shape[:2]
    avg2 = np.mean(points_2, axis=0)

    print(avg1, avg2)

    confidence = []

    # 1 left and 2 right
    confidence.append((avg1[0]/width1)*(1 - (avg2[0]/width2)))

    # 1 right and 2 left
    confidence.append((avg2[0]/width2)*(1 - (avg1[0]/width1)))

    # 1 up and 2 down
    confidence.append((avg1[1]/height1)*(1 - (avg2[1]/height2)))

    # 1 down and 2 up
    confidence.append((avg2[1]/height2)*(1 - (avg1[1]/height1)))

    max_ind = np.argmax(confidence)

    print(confidence)

    if max_ind == 0:
        return 'left', 'right', confidence[max_ind]

    if max_ind == 1:
        return 'right', 'left', confidence[max_ind]

    if max_ind == 2:
        return 'up', 'down', confidence[max_ind]

    return 'down', 'up', confidence[max_ind]


ARGS = (get_args())
SCALING = ARGS['scale']

print("[INFO] loading images...")
IMAGE_GRP_FOLDERS = sorted(os.listdir(ARGS["images"]))
print('[INFO] Found following image groups -', IMAGE_GRP_FOLDERS)

MATCHER = create_matcher()

for image_grp in IMAGE_GRP_FOLDERS:
    final_image_name = image_grp+'.jpg'
    print('[INFO] working on group - '+image_grp)
    images = load_images(ARGS['images']+'/'+image_grp)
    print("[INFO] Number of components = ", len(images))
    if SCALING != 100:
        images = [cv2.resize(x, (((x.shape[1])*SCALING)//100,
                                 (x.shape[0]*SCALING)//100)) for x in images]
    grayscale_imgs = [convert_to_grayscale(x) for x in images]
    described_imgs = [get_descriptors(x) for x in grayscale_imgs]
    # order = [i for i in range(len(images))]
    fin_hom = {}
    pair_hom = {}
    # for i in images:
    #     plt.imshow(i)
    #     plt.show()

    dummy = None
    for (ind1, ind2) in itertools.combinations(range(len(images)), 2):
        if ind1 >= ind2:
            continue

        print("Ongoing: ", (ind1, ind2))
        img1, img1_grayscale, img1_described = images[ind1], grayscale_imgs[ind1], described_imgs[ind1]
        img2, img2_grayscale, img2_described = images[ind2], grayscale_imgs[ind2], described_imgs[ind2]

        (kp1, dsc1), (kp2, dsc2) = img1_described, img2_described
        avg_distance, points1, points2, matches = get_matches(
            kp1, dsc1, kp2, dsc2)
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        pair_hom[(ind1, ind2)] = (h, avg_distance, matches)
        # loc1, loc2, conf = get_relative_position(img1, points1, img2, points2)
        # print(loc1, loc2, conf)
        display_image_with_matches(img1, kp1, img2, kp2, matches)

    fin_hom[0] = np.identity(pair_hom[(0, 1)][0].shape[0])

    visited = [0]
    selected = [(0, 0)]
    for ind in range(len(images)-1):
        best_pair = (ind, ind)
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
        selected.append((non_vis, vis))
        # print(best_pair)
        # print(visited)
        # print(fin_hom)

        if (non_vis, vis) in pair_hom:
            fin_hom[non_vis] = np.dot(
                fin_hom[vis], pair_hom[(non_vis, vis)][0])
        else:
            fin_hom[non_vis] = np.dot(
                fin_hom[vis], np.linalg.inv(pair_hom[(vis, non_vis)][0]))

    # height, width, channels = images[0].shape
    # fwidth = width
    # fheight = height
    # base = copy.deepcopy(images[0])
    # cum_trans = np.identity(pair_hom[(0, 1)][0].shape[0])
    # trans[2][2] = 0
    left_e = 0
    rot_r = 0
    right_e = 0
    rot_l = 0
    order = [0]
    centre_ind = 0
    for ind in range(1, len(selected)):
        if (selected[ind][1] == left_e) and (rot_r == 1):
            centre_ind -= 1
            rot_r = 0
            order.insert(0, selected[ind][0])
            centre_ind += 1
        elif (selected[ind][1] == right_e) and (rot_l == 1):
            centre_ind += 1
            rot_l = 0
            order.append(selected[ind][0])
        elif selected[ind][1] == left_e:
            if rot_l == 1:
                rot_r = 0
                rot_l = 0
            else:
                rot_r = 1
            order.insert(0, selected[ind][0])
            centre_ind += 1
        elif selected[ind][1] == right_e:
            if rot_r == 1:
                rot_l = 0
                rot_r = 0
            else:
                rot_l = 1
            order.append(selected[ind][0])
        else:
            print("Whoops!!! This shouldn't happen this way bro!!!!!")
            print("selected: ", selected)
            print("Trouble makers: ", selected[ind])
            # exit(0)
        left_e = order[0]
        right_e = order[len(order)-1]

        print("Order of images: ", order)

    print("Final Centre ID and order: ", centre_ind, " ", order)
    if order[centre_ind] != 0:
        for i in range(len(images)):
            fin_hom[i] = np.dot(
                pair_hom[(0, order[centre_ind])][0], fin_hom[i])

    # if len(images)%2 == 0:
    #     if centre_ind < len(images)/2:
    #         centre_ind += 1
    #     else:
    #         centre_ind -= 1

    fin_order = []
    for i in range(len(images)):
        if centre_ind+i < len(order):
            fin_order.append(order[centre_ind+i])
        if i != 0 and centre_ind-i >= 0:
            fin_order.append(order[centre_ind-i])

    print(fin_order)

    height, width, channels = images[fin_order[0]].shape
    fwidth = width
    fheight = height
    base = copy.deepcopy(images[fin_order[0]])
    cum_trans = np.identity(pair_hom[(0, 1)][0].shape[0])

    for i in fin_order[1:]:
        base, trans = combine_images(
            base, images[i], np.dot(cum_trans, fin_hom[i]))
        cum_trans = trans.dot(cum_trans)
        plt.imshow(base)
        plt.show()
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        base = base[y:y + h, x:x + w]
