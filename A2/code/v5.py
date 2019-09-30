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
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t_val = [-xmin, -ymin]
    h_translation = np.array(
        [[1, 0, t_val[0]], [0, 1, t_val[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(
        img2_loc, h_translation.dot(h_val), (xmax-xmin, ymax-ymin))

    plt.imshow(result)
    plt.show()
    img1_loc = np.array(img1_loc)
    img1 = np.sum(img1_loc, axis=2)
    indices1 = np.argwhere((img1 != 0))
    indices = indices1 + [t_val[1], t_val[0]]
    result[tuple(zip(*indices))] = img1_loc[tuple(zip(*indices1))]

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


def get_hom(a, b):
    ima = convert_to_grayscale(a)
    imb = convert_to_grayscale(b)
    desca = get_descriptors(ima)
    descb = get_descriptors(imb)
    (kp1, dsc1), (kp2, dsc2) = descb, desca
    avg_distance, points1, points2, matches = get_matches(kp1, dsc1, kp2, dsc2)
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    display_image_with_matches(imb, kp1, ima, kp2, matches)
    return h


def correct(base):
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    base = base[y:y + h, x:x + w]
    return base


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
    fin_hom = {}
    pair_hom = {}

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
        # display_image_with_matches(img1, kp1, img2, kp2, matches)

    # fin_hom[0] = np.identity(pair_hom[(0, 1)][0].shape[0])

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

                    if pair_hom[(ind11, ind21)][1] < min_distance or (len(pair_hom[(ind11, ind21)][2]) > max_num_matches and pair_hom[(ind11, ind21)][1] == min_distance):
                        best_pair = (ind11, ind21)
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

        # if (non_vis, vis) in pair_hom:
        #     fin_hom[non_vis] = np.dot(
        #         fin_hom[vis], pair_hom[(non_vis, vis)][0])
        # else:
        #     fin_hom[non_vis] = np.dot(
        #         fin_hom[vis], np.linalg.inv(pair_hom[(vis, non_vis)][0]))

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
            order.insert(order.index(selected[ind][1]), selected[ind][0])
            print("Whoops!!! This shouldn't happen this way bro!!!!!")
            print("selected: ", selected)
            print("Trouble makers: ", selected[ind])
            # exit(0)
        left_e = order[0]
        right_e = order[len(order)-1]

        print("Order of images: ", order)

    # centre_ind -= 1
    # if len(images)%2 == 0:
    #     if centre_ind < len(images)/2:
    #         centre_ind += 1
    #     else:
    #         centre_ind -= 1

    if len(images) <= 3:
        print("Final Centre ID and order: ", centre_ind, " ", order)
        fin_order = []
        for i in range(len(images)):
            if centre_ind+i < len(order):
                fin_order.append(order[centre_ind+i])
            if i != 0 and centre_ind-i >= 0:
                fin_order.append(order[centre_ind-i])

            if (centre_ind+i >= len(order)) and (centre_ind-i < 0):
                break
        print(fin_order)

        fin_hom[centre_ind] = np.identity(pair_hom[(0, 1)][0].shape[0])

        for i in range(1, len(fin_order)):
            if i < 3:
                if (fin_order[i], fin_order[0]) in pair_hom:
                    fin_hom[fin_order[i]] = (
                        pair_hom[(fin_order[i], fin_order[0])][0])
                else:
                    fin_hom[fin_order[i]] = np.linalg.inv(
                        pair_hom[(fin_order[0], fin_order[i])][0])
            else:
                if len(fin_order) > i+1 and i % 2 == 1:
                    if (fin_order[i], fin_order[i-2]) in pair_hom:
                        fin_hom[fin_order[i]] = np.dot(
                            fin_hom[fin_order[i-2]], pair_hom[(fin_order[i], fin_order[i-2])][0])
                        # fin_hom[fin_order[i]] = np.dot(pair_hom[(fin_order[i], fin_order[i-2])][0], fin_hom[fin_order[i-2]])
                    else:
                        fin_hom[fin_order[i]] = np.dot(
                            fin_hom[fin_order[i-2]], np.linalg.inv(pair_hom[(fin_order[i-2], fin_order[i])][0]))
                        # fin_hom[fin_order[i]] = np.dot(np.linalg.inv(pair_hom[(fin_order[i-2], fin_order[i])][0]), fin_hom[fin_order[i-2]])
                elif i % 2 == 1:
                    if (fin_order[i], fin_order[i-1]) in pair_hom:
                        fin_hom[fin_order[i]] = np.dot(
                            fin_hom[fin_order[i-1]], pair_hom[(fin_order[i], fin_order[i-1])][0])
                        # fin_hom[fin_order[i]] = np.dot(pair_hom[(fin_order[i], fin_order[i-1])][0], fin_hom[fin_order[i-1]])
                    else:
                        fin_hom[fin_order[i]] = np.dot(
                            fin_hom[fin_order[i-1]], np.linalg.inv(pair_hom[(fin_order[i-1], fin_order[i])][0]))
                        # fin_hom[fin_order[i]] = np.dot(np.linalg.inv(pair_hom[(fin_order[i-1], fin_order[i])][0]), fin_hom[fin_order[i-1]])
                else:
                    if (fin_order[i], fin_order[i-2]) in pair_hom:
                        fin_hom[fin_order[i]] = np.dot(
                            fin_hom[fin_order[i-2]], pair_hom[(fin_order[i], fin_order[i-2])][0])
                        # fin_hom[fin_order[i]] = np.dot(pair_hom[(fin_order[i], fin_order[i-2])][0], fin_hom[fin_order[i-2]])
                    else:
                        fin_hom[fin_order[i]] = np.dot(
                            fin_hom[fin_order[i-2]], np.linalg.inv(pair_hom[(fin_order[i-2], fin_order[i])][0]))
                        # fin_hom[fin_order[i]] = np.dot(np.linalg.inv(pair_hom[(fin_order[i-2], fin_order[i])][0]), fin_hom[fin_order[i-2]])

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
    else:
        my_im = []
        for i in range(len(images)):
            my_im.append(images[order[i]])
        # my_im = images
        i = len(my_im)-1
        j = 0
        final = None
        cum_transi = np.identity(pair_hom[(0, 1)][0].shape[0])
        cum_transj = np.identity(pair_hom[(0, 1)][0].shape[0])
        while True:
            if i <= j:
                break
            if i-j >= 3:
                hom1 = get_hom(my_im[i-1], my_im[i])
                hom2 = get_hom(my_im[j+1], my_im[j])
                base1, trans1 = combine_images(my_im[i-1], my_im[i], hom1)
                base2, trans2 = combine_images(my_im[j+1], my_im[j], hom2)
                # cum_transi = trans1.dot(cum_transi)
                # cum_transj = trans2.dot(cum_transj)
                base1 = correct(base1)
                base2 = correct(base2)

                my_im.pop(len(my_im)-1)
                my_im.pop(len(my_im)-1)
                my_im.append(base1)
                my_im.pop(0)
                my_im.pop(0)
                my_im.insert(0, base2)
                i = len(my_im)-1
                plt.imshow(base1)
                plt.show()
                plt.imshow(base2)
                plt.show()
            elif i-j == 2:
                hom1 = get_hom(my_im[i-1], my_im[i])
                base1, trans1 = combine_images(my_im[i-1], my_im[i], hom1)
                base1 = correct(base1)
                plt.imshow(base1)
                plt.show()
                # print(hom2)
                hom2 = get_hom(base1, my_im[j])
                if hom2 is not None:
                    base2, trans2 = combine_images(base1, my_im[j], hom2)
                else:
                    base2 = base1
                base2 = correct(base2)
                final = base2
                plt.imshow(base2)
                plt.show()
                i = j
                cv2.imwrite(ARGS["output"]+'/'+image_grp+'.jpg', final)
            elif i-j == 1:
                print((len(my_im), i))
                plt.imshow(my_im[i])
                plt.show()
                plt.imshow(my_im[i-1])
                plt.show()
                hom1 = get_hom(my_im[i], my_im[i-1])
                base1, trans1 = combine_images(my_im[i], my_im[i-1], hom1)
                base1 = correct(base1)
                final = base1
                # i = j
                plt.imshow(base1)
                plt.show()
                cv2.imwrite(ARGS["output"]+'/'+image_grp+'_1.jpg', final)

                hom1 = get_hom(my_im[i-1], my_im[i])
                base1, trans1 = combine_images(
                    my_im[i-1], my_im[i], np.dot(cum_transi, hom1))
                base1 = correct(base1)
                final = base1
                i = j
                plt.imshow(base1)
                plt.show()
                cv2.imwrite(ARGS["output"]+'/'+image_grp+'_0.jpg', final)
