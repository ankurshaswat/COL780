"""
All functions combined
"""
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

HARRIS = False
FEATURE_EXTRACTOR = 'brisk'  # 'sift'|'surf'|'brisk'|'orb'
FEATURE_MATCHER = 'bf'  # 'bf'|'knn'
NUM_GOOD_MATCHES = 200
LOWES_RATIO = 0.75
SCALING = 20  # Percent scale down
# MARKERS_PATHS = ['../markers/0.png', '../markers/1.png', '../markers/2.png']
MARKERS_PATHS = ['../markers/0.png', '../markers/2.png']


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


MATCHER = create_matcher()


def average(lst):
    """
    Average x.distance from a list of x.
    """
    sum_ = 0
    for i in lst:
        sum_ += i.distance
    return sum_ / len(lst)


def scale_image(img):
    """
    Scale down images
    """
    img = cv2.resize(img, ((
        (img.shape[1])*SCALING)//100, (img.shape[0]*SCALING)//100))
    return img


def load_image(image_path, scale=True):
    """
    Load image
    """
    image = cv2.imread(image_path)
    if SCALING != 100 and scale:
        image = scale_image(image)
    return image


def convert_to_grayscale(image):
    """
    Convert images from RGB to Grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_harris_keypoints(gray):
    """
    Returns corners found using harris corner detection algorithm with sub pixel accuracy.
    """
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    _, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    _, _, _, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(
        centroids), (5, 5), (-1, -1), criteria)
    keypoints = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in corners]

    return keypoints


def get_descriptors(gray_image):
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

    if HARRIS:
        kps = get_harris_keypoints(gray_image)
        keypoints, des = descriptor.compute(gray_image, kps)
        return (keypoints, des)

    return descriptor.detectAndCompute(gray_image, None)


def get_matches(kp1_loc, dsc1_loc, kp2_loc, dsc2_loc):
    """
    Get the matching descriptors according to feature matcher.
    """

    if FEATURE_MATCHER == 'bf':
        raw_matches = MATCHER.match(dsc1_loc, dsc2_loc)
        raw_matches.sort(key=lambda x: x.distance)
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


def display_image(img):
    """
    Show an image
    """
    plt.imshow(img)
    plt.show()


def draw_harris_kps(img):
    """
    Draw an image and its harris keypoints
    """

    gray_img = convert_to_grayscale(img)
    kps = get_harris_keypoints(gray_img)
    img_new = cv2.drawKeypoints(img, kps, None, color=(255, 0, 0))
    return img_new


def display_image_with_matches(img1_loc, kps1_loc, img2_loc, kps2_loc, matches_loc):
    """
    Display 2 images and their keypoints connected by lines side by side
    """
    img = cv2.drawMatches(img1_loc, kps1_loc, img2_loc, kps2_loc, matches_loc,
                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    plt.show()


def get_hom(img1_descriptors, img2):
    """
    Get homography between 2 images
    """
    imb = convert_to_grayscale(img2)
    desca = img1_descriptors
    descb = get_descriptors(imb)
    (kp1, dsc1), (kp2, dsc2) = desca, descb
    avg_dist, points1, points2, matches = get_matches(kp1, dsc1, kp2, dsc2)
    hom, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    return hom, matches, avg_dist, kp2


def get_homography_from_corners(corners, ref_image):
    """
    Get homography using corners of reference image
    """
    pts_dst = np.array([[corners[0][0][0], corners[0][0][1]],
                        [corners[0][1][0], corners[0][1][1]],
                        [corners[0][2][0], corners[0][2][1]],
                        [corners[0][3][0], corners[0][3][1]]])
    pts_src = ref_image

    homography, status = cv2.findHomography(pts_src, pts_dst)
    return homography, status


def get_matrix(camera_params, homography, translate=None):
    """
    Using camera params and homography matrix get projection matrix
    """
    r_t = np.dot(np.linalg.inv(camera_params), homography)
    r_1 = r_t[:, 0]
    r_2 = r_t[:, 1]
    t_vec = r_t[:, 2]
    norm = math.sqrt(np.linalg.norm(r_1, 2) * np.linalg.norm(r_2, 2))
    r_1 = r_1/norm
    r_2 = r_2/norm
    t_vec = t_vec/norm
    c_val = r_1+r_2
    p_val = np.cross(r_1, r_2)
    d_val = np.cross(c_val, p_val)
    r1_ = (1/math.sqrt(2))*(c_val/np.linalg.norm(c_val, 2) +
                            d_val/np.linalg.norm(d_val, 2))
    r2_ = (1/math.sqrt(2))*(c_val/np.linalg.norm(c_val, 2) -
                            d_val/np.linalg.norm(d_val, 2))
    r3_ = np.cross(r1_, r2_)
    proj = np.stack((r1_, r2_, r3_, t_vec)).T

    if translate is None:
        return np.dot(camera_params, proj)

    return np.dot(camera_params, np.dot(translate, proj))


def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    height, width, _ = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array(
            [[p[0] + width / 2, p[1] + height / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape((-1, 1, 3)), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def get_camera_params():
    """
    Return a default camera parameter matrix
    """
    return np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])


def load_ref_images(num_img=2):
    """
    Load visual markers reference images
    """

    ref_imgs = []

    for marker_path in MARKERS_PATHS:
        ref_imgs.append(load_image(marker_path, False))

    ref_imgs = ref_imgs[:num_img]

    ref_descriptors = []

    for img in ref_imgs:
        img_grayscale = convert_to_grayscale(img)
        ref_descriptors.append(get_descriptors(img_grayscale))

    return ref_imgs, ref_descriptors


def find_homographies(reference_descriptors, frame):
    """
    Find homography for each reference image
    """
    match_data = []  # (homography,matches,avg_dist)

    for descriptor in reference_descriptors:
        match_data.append(get_hom(descriptor, frame))

    return match_data


def draw_rectangle(homography, ref_img, frame, color=255):
    """
    Draw rectangles around each found visual marker
    """
    height, width, _ = ref_img.shape
    pts = np.array([[0, 0], [0, height - 1], [width - 1, height - 1],
                    [width - 1, 0]], dtype='float32').reshape((-1, 1, 2))
    dst = cv2.perspectiveTransform(pts, homography)
    frame = cv2.polylines(
        frame, [np.int32(dst)], True, color, 3, cv2.LINE_AA)
    return frame


def calculate_dist(kp1, matches1, kp2, matches2):
    """
    Function to calculate distance between two planes after projecting using homographies.
    """
    minimum = min(len(matches1), len(matches2))
    points1 = np.zeros((minimum, 2), dtype=np.float32)
    points2 = np.zeros((minimum, 2), dtype=np.float32)

    for i, match in enumerate(matches1[:minimum]):
        points1[i, :] = kp1[match.trainIdx].pt

    for i, match in enumerate(matches2[:minimum]):
        points2[i, :] = kp2[match.trainIdx].pt

    dist = np.average(points2-points1, axis=0)
    return dist
