import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

algo = 'KNN' # MOG2 or KNN
window = 3
frame_seq_path = '../videos/1.mp4'

if algo=='MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(frame_seq_path))
if not capture.isOpened:
    print('Unable to open: ' + frame_seq_path)
    exit(0)

glb_frames = []
local_frames = []
while True:
    ret, frame = capture.read()
    if frame is None:
        print('Frames over')
        break
    
    fgMask = backSub.apply(frame)

    # ------ Denoising part. Uncomment to see ----------------
    # glb_frames.append(fgMask)
    # if len(local_frames) >= window:
    #     local_frames.remove(local_frames[0])
    #     local_frames.append(fgMask)
    # else:
    #     if len(local_frames)%2 == 1 and (len(glb_frames)-len(local_frames))==2:
    #         local_frames.insert(0,glb_frames[-1-len(local_frames)])
    #         local_frames.append(fgMask)
    #     elif len(local_frames) > 0:
    #         local_frames.remove(local_frames[0])
    #         local_frames.append(fgMask)
    #     else:
    #         local_frames.append(fgMask)

    # actual_window_size = len(local_frames)
    # cv.fastNlMeansDenoisingMulti(local_frames, int(actual_window_size/2), actual_window_size, None, 4, 7, 21)

    # Single wala denoising. Try yourself by uncommenting
    # fgMask = cv.fastNlMeansDenoising(fgMask, None, 4, 7, 21)

    # Uncomment if you want to see fastNlMeansDenoisingMulti and comment the below im.show line
    # cv.imshow('FG Mask', local_frames[int(actual_window_size/2)])
    # --------------------------------------------------------

    # Window ke saath kuch karta hai 
    # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    # cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))    


    edges = cv.Canny(fgMask,50,100,apertureSize = 3)

    lines = cv.HoughLines(edges,1,np.pi/180,100)
    # # print(lines)
    # # print(type(lines) is None)
    old_fg = copy.deepcopy(fgMask)
    # print(fgMask.any()> 0)
    transformed_lines = []
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            transformed_lines.append([(x1,y1), (x2,y2)])
            cv.line(fgMask,(x1,y1),(x2,y2),(255,0,0),4)
            cv.line(frame,(x1,y1),(x2,y2),(0,0,255),4)
            # print((a-old_fg).any()>0)
        # print((fgMask-old_fg).any() > 0)

    # Trying skeleton wala stuff
    # size = np.size(fgMask)
    # print(size)
    # skel = np.zeros(fgMask.shape,np.uint8)
    # element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    # done = False
    # while( not done):
    #     eroded = cv.erode(fgMask,element)
    #     temp = cv.dilate(eroded,element)
    #     temp = cv.subtract(fgMask,temp)
    #     skel = cv.bitwise_or(skel,temp)
    #     fgMask = eroded.copy()
    #     # print(old_fg)
    #     print((fgMask-old_fg).any()>0)
    #     zeros = size - cv.countNonZero(fgMask)
    #     print("Let's see: ", zeros, " ", cv.countNonZero(fgMask))
        
    #     if zeros==size:
    #         done = True

    # print("YOOOOOOOOOOO")
    # # Window Size Normal
    cv.namedWindow("FG Mask", cv.WINDOW_NORMAL)
    # cv.namedWindow("Frame", cv.WINDOW_NORMAL)

    # cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    # cv.imshow('FG Mask', fgMask-old_fg)
    # cv.imshow('FG Mask', skel)
    # cv.imshow('FG Mask', edges)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        print('Exit from keyboard')
        break