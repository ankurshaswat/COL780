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
MARKERS_PATHS = ['../../markers/0.png', '../../markers/2.png']
REF_IMAGE1 = np.array([[0, 0],
                      [0, 700],
                      [700, 700],
                      [700, 0]], dtype=np.float32)
REF_IMAGE2 = np.array([[0, 0],
                      [0, 700],
                      [700, 700],
                      [700, 0]], dtype=np.float32)

REF_IMAGE3 = np.array([[0, 0],
                      [0, 700],
                      [700, 700],
                      [700, 0]], dtype=np.float32)

ROT = np.array([[0,1],[3,2]])

def do_everything(FRAMED, REF_IMAGES, OBJ, CAM_MAT, REACHED_X, REACHED_Y):
    FRAME = convert_to_grayscale(FRAMED)
    # ret, thresh = cv2.threshold(FRAME,127,255,cv2.THRESH_BINARY)        
    FRAME = cv2.GaussianBlur(FRAME,(5,5),0)
    thresh = cv2.adaptiveThreshold(FRAME,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("thresh", thresh)
    # print(thresh)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    area_contour = [contour for contour in contours if cv2.contourArea(contour) >= 500]
    rects = [cv2.minAreaRect(i) for i in area_contour]
    boxes = [np.int0(cv2.boxPoints(i)) for i in rects]
    warped = [four_point_transform(FRAME, box) for box in boxes]
    selected_markers = compare_markers(warped, REF_IMAGES)

    # cv2.imshow("0", selected_markers[0][0])
    # cv2.imshow("1", selected_markers[1][0])
    for sel in selected_markers:
        try:
            if sel[3] <= 4:
                cv2.drawContours(FRAMED,[boxes[sel[2]]],-1,(0,0,255),3)
        except:
            pass

    MATCH_DATA = [None, None]
    try:
        if selected_markers[0][3] <= 4:
            ROT3 = np.rot90(ROT)
            REF_IMAGE3[0] = REF_IMAGE1[ROT3[0,0]]
            REF_IMAGE3[1] = REF_IMAGE1[ROT3[0,1]]
            REF_IMAGE3[2] = REF_IMAGE1[ROT3[1,1]]
            REF_IMAGE3[3] = REF_IMAGE1[ROT3[1,0]]
            h1 = get_homography_from_corners(boxes[selected_markers[0][2]], REF_IMAGE3)[0]
        
        h2 = get_homography_from_corners(boxes[selected_markers[1][2]], REF_IMAGE2)[0]
        MATCH_DATA = [h1, h2]
    except:
        pass

    if MATCH_DATA[0] is not None and MATCH_DATA[1] is not None:
        HOMOGRAPHY1 = MATCH_DATA[0]
        HOMOGRAPHY2 = MATCH_DATA[1]
        corner1 = boxes[selected_markers[0][2]]
        corner2 = boxes[selected_markers[1][2]]

        PROJ_MAT1 = get_matrix(CAM_MAT, HOMOGRAPHY1)

        DIST = calculate_dist_corners(corner1, corner2)
        DIST_X = DIST[0]
        DIST_Y = DIST[1]

        STEP_X = DIST_X/10
        STEP_Y = DIST_Y/10

        if abs(REACHED_X) >= abs(DIST_X) or abs(REACHED_Y) >= abs(DIST_Y):
            REACHED_X = 0
            REACHED_Y = 0
        else:
            REACHED_X += STEP_X
            REACHED_Y += STEP_Y

        TRANS = np.array(
            [[1, 0, REACHED_X], [0, 1, REACHED_Y], [0, 0, 1]])
        PROJ_MAT = np.dot(TRANS, PROJ_MAT1)
        FRAMED = render(FRAMED, OBJ, PROJ_MAT, REF_IMAGE1, False)
    return REACHED_X, REACHED_Y

def get_middle(arr):
    n = np.array(arr.shape) / 2.0
    n_int = np.int0(n)
    if n[0] % 2 == 1 and n[1] %2 == 1:
        return arr[n_int[0], n_int[1]]
    elif n[0] %2 == 0 and n[1] %2 == 0:
        return np.average(arr[n_int[0]:n_int[0] + 2, n_int[1]:n_int[1] + 2])
    elif n[0] %2 == 1 and n[1] %2 == 0:
        return np.average(arr[n_int[0], n_int[1]:n_int[1]+2])
    else:
        return np.average(arr[n_int[0]:n_int[0]+2, n_int[1]])

def get_binary(images, divsions):
    # threshed = [cv2.threshold(warp, 127, 255, cv2.THRESH_BINARY)[1]/255 for warp in images]
    threshed = [cv2.GaussianBlur(thresh,(5,5),0) for thresh in images]
    # threshed = images
    # threshed = [cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) for warp in threshed]
    threshed = [cv2.threshold(warp, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]/255 for warp in threshed]
    mats = []
    for thr in threshed:
        shap = thr.shape
        jump = np.int0(np.array(shap)/divsions)
        mat = np.zeros([divsions,divsions])
        for i in range(divsions):
            for j in range(divsions):
                try:
                    avg = get_middle(thr[i*jump[0]:(i+1)*jump[0], j*jump[1]:(j+1)*jump[1]])
                except:
                    avg = 0
                # avg = np.average(thr[i*jump[0]:(i+1)*jump[0], j*jump[1]:(j+1)*jump[1]])
                if avg >= 0.5:
                    mat[i][j] = 1
                else:
                    mat[i][j] = 0
        mats.append(mat)
    return mats


def compare_markers(matrices, ref_images):
    thresh = get_binary(matrices, 7)
    refs = get_binary([convert_to_grayscale(ref) for ref in ref_images], 7)
    ret = []
    for ref in refs:
        mini = np.sum(np.absolute(refs[0]-thresh[0]))
        rot = (thresh[0], ref, 0, mini)
        for thr in range(0, len(thresh)):
            for i in range(4):
                su = np.sum(np.absolute(np.rot90(ref, i)-thresh[thr]))
                # print("su, mini ", su, mini)
                if su < mini:
                    if len(ret) > 0 and ret[0][2] != thr:
                        mini = su
                        rot = (thresh[thr], i, thr, mini)
                    elif len(ret) == 0:
                        mini = su
                        rot = (thresh[thr], i, thr, mini)
        ret.append(rot)
    return ret

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    return warped

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
    if len(lst) == 0:
        return 10
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
    pts_dst = np.array([[corners[0][0], corners[0][1]],
                        [corners[1][0], corners[1][1]],
                        [corners[2][0], corners[2][1]],
                        [corners[3][0], corners[3][1]]])
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
    old_img = img.copy()
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    height, width = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        # print(points.shape)
        points = np.array(
            [[p[0] + width / 2, p[1] + height / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape((-1, 1, 3)), projection)
        imgpts = np.int32(dst)
        mini_img = np.min(imgpts, axis=0)[0]
        maxi_img = np.max(imgpts, axis=0)[0]
        mini_canvas = [0, 0]
        maxi_canvas = [img.shape[0], img.shape[1]]

        if (mini_img[0] < mini_canvas[0]) or (mini_img[1] < mini_canvas[1]) or (maxi_img[0] > maxi_canvas[0]) or (maxi_img[1] > maxi_canvas[1]):
            return old_img

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
    return frame, dst


def calculate_dist_matches(kp1, matches1, kp2, matches2):
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

    # print("mats: ", points2-points1)
    a = np.average(points2-points1, axis=0)
    # print("average: ", a)
    return a


def calculate_dist_corners(corner1, corner2):
    """
    Function to calculate distance between two planes after projecting using homographies.
    """
    dst = np.average(corner2-corner1, axis=0)
    return dst


def display_image_with_matched_keypoints(img1_loc, kps1_loc):
    """
    Display 2 images and their keypoints side by side
    """
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                 figsize=(20, 8), constrained_layout=False)
    return cv2.drawKeypoints(img1_loc, kps1_loc, None, color=(255, 0, 0))


def init_game(size_window):
    """
    Init game of ping pong.
    """
    print(size_window)
    game_obj = {
        'pos': (size_window[1]/2, size_window[0]/2),
        'velocity': (10, 10),
        'rudder1_pos': (0.05*size_window[1], size_window[0]/2),
        'rudder2_pos': (0.95*size_window[1], 1.5*size_window[0]/2),
        'score1': 0,
        'score2': 0,
    }

    return game_obj


def draw_game(frame, size, game_obj):
    """
    Draw game using game_object on frame.
    """
    pos_float = game_obj['pos']
    pos_int = ((int)(pos_float[0]), (int)(pos_float[1]))
    frame = cv2.circle(frame, pos_int, 15, (0, 0, 255), -1)
    frame = cv2.rectangle(frame, (15, 15), (640-15, 480-15), (0, 255, 0), 1)

    rudder_len = 50
    rudder_thickness = 2

    player1_bat_center = game_obj['rudder1_pos']
    player2_bat_center = game_obj['rudder2_pos']

    p1_start = (int(
        player1_bat_center[0]-rudder_thickness), int(player1_bat_center[1]-rudder_len))
    p1_end = (
        int(player1_bat_center[0]+rudder_thickness), int(player1_bat_center[1]+rudder_len))

    p2_start = (
        int(player2_bat_center[0]-rudder_thickness), int(player2_bat_center[1]-rudder_len))
    p2_end = (
        int(player2_bat_center[0]+rudder_thickness), int(player2_bat_center[1]+rudder_len))

    frame = cv2.rectangle(frame, p1_start, p1_end, (0, 255, 0), -1)
    frame = cv2.rectangle(frame, p2_start, p2_end, (0, 255, 0), -1)

    size_x = size[1]
    size_y = size[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, size_y-10)
    bottomRightCornerOfText = (size_x-30, size_y-10)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(frame, str(game_obj['score1']),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.putText(frame, str(game_obj['score2']),
                bottomRightCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return frame


def update_game(game_obj, size, center1, center2):
    """
    Update game state
    """
    new_game_obj = game_obj.copy()

    if center1 is not None:
        new_game_obj['rudder1_pos'] = center1
    if center2 is not None:
        new_game_obj['rudder2_pos'] = center2

    # Check if hitting corner
    init_vel = new_game_obj['velocity'] 
    if new_game_obj['pos'][1] >= 480-15 or new_game_obj['pos'][1] <= 15:
        new_game_obj['velocity'] = (init_vel[0], -1*init_vel[1])
    if new_game_obj['pos'][0] >= 640-15:
        new_game_obj['pos'] = (size[1]/2, size[0]/2)
        new_game_obj['velocity'] = (-1.05*abs(new_game_obj['velocity'][0]),
                                    1.05*abs(new_game_obj['velocity'][1]))
        new_game_obj['score1'] += 1
    elif new_game_obj['pos'][0] <= 15:
        new_game_obj['pos'] = (size[1]/2, size[0]/2)
        new_game_obj['score2'] += 1
        new_game_obj['velocity'] = (1.05*abs(new_game_obj['velocity'][0]),
                                    -1.05*abs(new_game_obj['velocity'][1]))
    elif 0 <= new_game_obj['pos'][0]-new_game_obj['rudder1_pos'][0] <= 17 and new_game_obj['rudder1_pos'][1]-(50+15) < new_game_obj['pos'][1] < new_game_obj['rudder1_pos'][1] + 50+15:
        new_game_obj['velocity'] = (-1*init_vel[0], init_vel[1])
    elif 0 <= new_game_obj['rudder2_pos'][0] - new_game_obj['pos'][0] <= 17 and new_game_obj['rudder2_pos'][1]-(50+15) < new_game_obj['pos'][1] < new_game_obj['rudder2_pos'][1]+(50+15):
        init_vel = new_game_obj['velocity']
        new_game_obj['velocity'] = (-1*init_vel[0], init_vel[1])

    new_game_obj['pos'] = (new_game_obj['pos'][0] + new_game_obj['velocity']
                           [0], new_game_obj['pos'][1] + new_game_obj['velocity'][1])

    # print(new_game_obj)
    return new_game_obj
