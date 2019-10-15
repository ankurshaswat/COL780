"""
Move object from one visual marker to another
"""

import sys

import cv2
import numpy as np

import obj_loader
from utils import (draw_rectangle, find_homographies, get_matrix, get_homography_from_corners,
                   load_ref_images, calculate_dist_matches, calculate_dist_corners, render, display_image_with_matched_keypoints)

if __name__ == "__main__":
    OBJ_PATH = sys.argv[1]

    OBJ = obj_loader.OBJ(OBJ_PATH, swapyz=True)
    REF_IMAGE1 = np.array([[0, 0],
                          [1000, 0],
                          [0, 1000],
                          [1000, 1000]], dtype=np.float32)
    REF_IMAGE2 = np.array([[0, 0],
                          [1000, 0],
                          [0, 1000],
                          [1000, 1000]], dtype=np.float32)
    # REF_IMAGES, REF_DSC = load_ref_images()
    # print(REF_DSC)
    VID_FEED = cv2.VideoCapture(-1)

    REACHED_X, REACHED_Y = 0, 0

    iterate = 0

    flag_g = True
    flag_b = True
    while True:
        box = None
        box_g = None
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        SIZE = FRAME.shape
        FOC_LEN = SIZE[1]
        CENTER = (SIZE[1]/2, SIZE[0]/2)

        CAM_MAT = np.array(
            [[FOC_LEN, 0, CENTER[0]],
             [0, FOC_LEN, CENTER[1]],
             [0, 0, 1]], dtype="double")
        hsv = cv2.cvtColor(FRAME, cv2.COLOR_BGR2HSV) 

        lowerGreen=np.array([33,80,40])
        upperGreen=np.array([102,255,255])
        maskGreen = cv2.inRange(hsv, lowerGreen, upperGreen)
        mask1 = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        maskGreen = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))        
        conts_g,h_g=cv2.findContours(maskGreen.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        maxw = 0
        maxh = 0
        ind = -1
        for i in range(len(conts_g)):
            x,y,w,h=cv2.boundingRect(conts_g[i])
            if w > maxw and h > maxh:
                maxh = h
                maxw = w
                ind = i

        if ind >= 0:
            if flag_g:
                rect_g = cv2.minAreaRect(conts_g[ind])
                box_g = cv2.boxPoints(rect_g)
                box_g = np.int0(box_g)
                last_contour_g = box_g
                flag_g = False

            conts_g = np.array([conts_g[ind]])
            area_g = cv2.contourArea(conts_g[0])        
            if area_g >= 500:
                rect_g = cv2.minAreaRect(conts_g[0])
                box_g = cv2.boxPoints(rect_g)
                box_g = np.int0(box_g)
                cv2.drawContours(FRAME,[box_g],0,(0,0,255),2)
                last_contour_g = box_g
                # cv2.drawContours(FRAME,conts_g,-1,(255,0,0),3)
            else:
                try:
                    cv2.drawContours(FRAME,last_contour_g,-1,(255,0,0),3)
                except:
                    pass     
        # Range for upper range
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
         
        # Generating the final mask to detect red color
        maskBlack = mask2
        mask1 = cv2.morphologyEx(maskBlack, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        maskBlack = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))        
        conts_b,h_b=cv2.findContours(maskBlack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        maxw = 0
        maxh = 0
        ind = -1
        for i in range(len(conts_b)):
            x,y,w,h=cv2.boundingRect(conts_b[i])
            if w > maxw and h > maxh:
                maxh = h
                maxw = w
                ind = i

        if ind >= 0:
            if flag_b:
                rect = cv2.minAreaRect(conts_b[ind])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                last_contour_b = box
                flag_b = False

            conts_b = np.array([conts_b[ind]])
            area_b = cv2.contourArea(conts_b[0])        
            if area_b >= 500:
                rect = cv2.minAreaRect(conts_b[0])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(FRAME,[box],0,(0,0,255),2)
                # print(box)
                # cv2.drawContours(FRAME,conts_b,-1,(255,0,0),3)
                last_contour_b = box
            else:
                try:
                    cv2.drawContours(FRAME,last_contour_b,-1,(255,0,0),3)
                except:
                    pass
        # cv2.imshow("masks",maskGreen+maskBlack)
        # cv2.imshow("frame",FRAME)
        # cv2.imshow("maskOpen",maskBlack)
        # cv2.waitKey(1)

        MATCH_DATA = [None, None]
        try:
            h1 = get_homography_from_corners(box, REF_IMAGE1)[0]
            h2 = get_homography_from_corners(box_g, REF_IMAGE2)[0]
            MATCH_DATA = [h1, h2]
        except:
            pass
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # print(MATCH_DATA)
        if MATCH_DATA[0] is not None and MATCH_DATA[1] is not None:
            HOMOGRAPHY1 = MATCH_DATA[0]
            HOMOGRAPHY2 = MATCH_DATA[1]
            corner1 = box
            corner2 = box_g

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
            PROJ_MAT1 = np.dot(TRANS, PROJ_MAT1)

            # FRAME = render(FRAME, OBJ, PROJ_MAT1, REF_IMAGES[0], False)
            FRAME = render(FRAME, OBJ, PROJ_MAT1, REF_IMAGE1, False)
            cv2.imshow("frame", FRAME)
        else:
            cv2.imshow("frame", FRAME)

    VID_FEED.release()
    cv2.destroyAllWindows()
