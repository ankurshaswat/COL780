"""
Ping pong game
"""

import sys

import cv2
import numpy as np

import obj_loader
from utils import (init_game, draw_game, update_game)

if __name__ == "__main__":
    # REF_IMAGES, REF_DSC = load_ref_images()
    VID_FEED = cv2.VideoCapture(-1)

    GAME_STATE = None

    flag_g = True
    flag_b = True
    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()
        elif GAME_STATE is None:
            SIZE = FRAME.shape
            GAME_STATE = init_game(SIZE)
            FOC_LEN = SIZE[1]
            CENTER = (SIZE[1]/2, SIZE[0]/2)
            CAM_MAT = np.array(
                [[FOC_LEN, 0, CENTER[0]],
                 [0, FOC_LEN, CENTER[1]],
                 [0, 0, 1]], dtype="double")

        # MATCH_DATA = find_homographies(REF_DSC, FRAME)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

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
                last_contour_b = box
            else:
                try:
                    cv2.drawContours(FRAME,last_contour_b,-1,(255,0,0),3)
                except:
                    pass


        # if MATCH_DATA[0][0] is not None and MATCH_DATA[1][0] is not None:

        #     HOMOGRAPHY1 = MATCH_DATA[0][0]
        #     HOMOGRAPHY2 = MATCH_DATA[1][0]
        #     MATCHES1 = MATCH_DATA[0][1]
        #     MATCHES2 = MATCH_DATA[1][1]
        #     KP1 = MATCH_DATA[0][3]
        #     KP2 = MATCH_DATA[1][3]

        #     FRAME, corner1 = draw_rectangle(HOMOGRAPHY1, REF_IMAGES[0], FRAME)
        #     FRAME, corner2 = draw_rectangle(
        #         HOMOGRAPHY2, REF_IMAGES[1], FRAME, 120)

        #     PROJ_MAT1 = get_matrix(CAM_MAT, HOMOGRAPHY1)

        #     # DIST = calculate_dist_matches(KP1, MATCHES1, KP2, MATCHES2)
        #     DIST = calculate_dist_corners(corner1, corner2)

        #     kp2 = [KP1[match.trainIdx] for match in MATCHES1]
        #     FRAME = display_image_with_matched_keypoints(FRAME, kp2)
        #     DIST_X = DIST[0]
        #     DIST_Y = DIST[1]

        #     STEP_X = DIST_X/10
        #     STEP_Y = DIST_Y/10

        #     if abs(REACHED_X) >= abs(DIST_X) or abs(REACHED_Y) >= abs(DIST_Y):
        #         REACHED_X = 0
        #         REACHED_Y = 0
        #     else:
        #         REACHED_X += STEP_X
        #         REACHED_Y += STEP_Y

        #     TRANS = np.array(
        #         [[1, 0, REACHED_X], [0, 1, REACHED_Y], [0, 0, 1]])
        #     PROJ_MAT1 = np.dot(TRANS, PROJ_MAT1)

        #     FRAME = render(FRAME, OBJ, PROJ_MAT1, REF_IMAGES[0], False)
        #     cv2.imshow("frame", FRAME)
        # else:
        #     cv2.imshow("frame", FRAME)
        # GAME_STATE['rudder1_pos'] = 
        # GAME_STATE['rudder2_pos']

        FRAME = draw_game(FRAME, SIZE, GAME_STATE)
        cv2.imshow("frame", FRAME)
        GAME_STATE = update_game(GAME_STATE, SIZE, [0.05*(FRAME.shape[1]), np.average(box, axis = 0)[1]], [0.95*(FRAME.shape[1]), np.average(box_g, axis = 0)[1]])

    VID_FEED.release()
    cv2.destroyAllWindows()
