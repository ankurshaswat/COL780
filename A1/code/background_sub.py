import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import math

algo = 'KNN' # MOG2 or KNN
window = 3
frame_seq_path = '../videos/1.mp4'
top_lines = 5

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
    # print(edges.any() > 0)
    # print([i for i in edges[100] if edges[100][i] > 0])
    indices = np.argwhere(edges>0)
    # if(len(indices) > 0):
        # print(indices[0][0])
    #     print(edges[indices[0][0], indices[0][1]])
    
    # new_edges = np.zeros(np.shape(edges))
    new_edges = copy.deepcopy(edges)
    # new_edges[new_edges[new_edges>=0]] = 0
    for i in range(edges.shape[0]):
        if len(indices) > 0:
            # print(indices[indices[:,0] == i])
            # print(np.average(indices[indices[:,0] == i], axis=0)[1])    
            new_edges[i,:] = 0
            pixel = np.average(indices[indices[:,0] == i], axis=0)[1]
            if not math.isnan(pixel):
                # print(type(np.int16(new_edges.shape[1]).item()))
                # print(type(pixel))
                pixel = np.int16(pixel).item()
                for x in range(max(0,pixel-3), min(np.int16(new_edges.shape[1]).item(), pixel+3)):
                    # new_edges[row][x] = 255
                    new_edges[i][int(pixel)] = 255

    # row = 0
    # for i in edges:
    #     pixel = 0
    #     count = 0
    #     for j in range(len(i)):
    #         if i[j]>0:
    #             pixel+=j
    #             count+=1
    #     if count > 0:
    #         pixel = int(pixel/count)
    #         for x in range(max(0,pixel-10), min(new_edges.shape[1], pixel+10)):
    #             new_edges[row][x] = 255
    #     row += 1

    # print(new_edges.any() > 0)

    # lines = cv.HoughLines(edges,1,np.pi/180,100)
    lines = cv.HoughLines(new_edges,1,np.pi/180,100)
    old_fg = copy.deepcopy(fgMask)
    # transformed_lines = []
    avg = [0,0,0,0]
    avg_try2 = [0,0,0,0]
    if lines is not None:
        print(len(lines))
        for single_line in lines[0:min(len(lines), top_lines)]:
            for rho,theta in single_line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                avg_try2[0] += x0;
                avg_try2[1] += y0;
                avg_try2[2] += rho;
                avg_try2[3] += theta;
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                avg[0] += x1;
                avg[1] += y1;
                avg[2] += x2;
                avg[3] += y2;
            # transformed_lines.append([(x1,y1), (x2,y2)])
            # cv.line(fgMask,(x1,y1),(x2,y2),(255,0,0),4)
            # cv.line(frame,(x1,y1),(x2,y2),(0,0,255),4)
            # print((a-old_fg).any()>0)
        # print((fgMask-old_fg).any() > 0)

    # --------------------------------------------------
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
    # -----------------------------------------------------

    # # Window Size Normal
    # cv.namedWindow("FG Mask", cv.WINDOW_NORMAL)
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)

    if lines is not None:
        num = min(len(lines), top_lines)
        avg_try2[0] = avg_try2[0]/num
        avg_try2[1] = avg_try2[1]/num
        a = np.cos(avg_try2[3]/num)
        b = np.sin(avg_try2[3]/num)
        x0 = a*rho
        y0 = b*rho

        x1 = int(avg_try2[0] + 1000*(-b))
        y1 = int(avg_try2[1] + 1000*(a))
        x2 = int(avg_try2[0] - 1000*(-b))
        y2 = int(avg_try2[1] - 1000*(a))

        x11 = int(x0 + 1000*(-b))
        y11 = int(y0 + 1000*(a))
        x22 = int(x0 - 1000*(-b))
        y22 = int(y0 - 1000*(a))
        # cv.line(frame,(x11,y11),(x22,y22),(255,0,0),4)
        # cv.line(frame,(x1,y1),(x2,y2),(255,0,0),4)
        cv.line(frame,(int(avg[0]/num),int(avg[1]/num)),(int(avg[2]/num),int(avg[3]/num)),(0,0,255),4)
        # cv.line(new_edges,(int(avg[0]/num),int(avg[1]/num)),(int(avg[2]/num),int(avg[3]/num)),(0,0,255),4)

    cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    # cv.imshow('FG Mask', fgMask-old_fg)
    # cv.imshow('FG Mask', skel)
    # cv.imshow('FG Mask', edges)
    # cv.imshow('FG Mask', new_edges)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        print('Exit from keyboard')
        break