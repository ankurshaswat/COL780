import math
import copy
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

VIDEO_PATH = '../videos/1.mp4'
OUT_VIDEO_PATH = '../out_videos/result_1.avi'
NUMBER_LINES_CUTOFF = 5
AXIS_SIZE = 3
LENGTH_PERCENTAGE_THRESH = 70

def find_median_axis(frame):
    indices = np.argwhere(edges > 0)

    averaged_median_axis = copy.deepcopy(edges)
    
    variance_nan = [np.var(indices[indices[:, 0] == i][:,1]) for i in range(frame.shape[0])]
    variance = [ variance_nan[i] if not math.isnan(variance_nan[i]) else 0 for i in range(len(variance_nan))]
 
    for i in range(edges.shape[0]):
        if len(indices) > 0 and variance[i]<2000 and variance[i]!=0:
            averaged_median_axis[i, :] = 0
            pixel = np.average(indices[indices[:, 0] == i], axis=0)[1]
            if not math.isnan(pixel):
                pixel = np.int16(pixel).item()
                for x in range(max(0, pixel-AXIS_SIZE), min(np.int16(averaged_median_axis.shape[1]).item(), pixel+AXIS_SIZE)):
                    averaged_median_axis[i][x] = 255
        else:
            averaged_median_axis[i, :] = 0

    return averaged_median_axis

# Get lines from Rho and Theta returned by Hough Transform using the bound y2
def get_line(lines,y2):
    avg = [0, 0, 0, 0]

    num = min(len(lines), NUMBER_LINES_CUTOFF)

	# Averaging over top NUMBER_LINES_CUTOFF most voted lines
    for rho, theta in lines[0:min(len(lines), NUMBER_LINES_CUTOFF)]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        x1 = x0 + (b*y0/a)
        x2 = x1 - (b*y2/a)
        y1 = 0

        avg[0] += x1
        avg[1] += y1
        avg[2] += x2
        avg[3] += y2

    return (int(avg[0]/num), int(avg[1]/num)), (int(avg[2]/num), int(avg[3]/num))

# Finding out the length of the line to be a median axis and not overflow
def approximate_split(frame):
    frame_copy = copy.deepcopy(frame)
    indices = np.argwhere(frame_copy>0)
 
    variance_nan = [np.var(indices[indices[:, 0] == i][:,1]) for i in range(frame.shape[0])]
    variance = [ variance_nan[i] if not math.isnan(variance_nan[i]) else 0 for i in range(len(variance_nan))]
 
    sum_ = np.sum(frame_copy>0,axis=1)
    cum_sum = np.cumsum(sum_)
    percentages = 100*cum_sum/cum_sum[-1]

    for i in range(percentages.shape[0]):
        if(percentages[i]>LENGTH_PERCENTAGE_THRESH):
            return i

    return 0

backSub = cv.createBackgroundSubtractorKNN()
backSub1 = cv.createBackgroundSubtractorMOG2()

vids = os.listdir("../videos/")
for video in vids:
	VIDEO_PATH = '../videos/'+video
	OUT_VIDEO_PATH = '../out_videos/result_'+ video.split('.')[0] +'.avi'
	capture = cv.VideoCapture(cv.samples.findFileOrKeep(VIDEO_PATH))
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter(OUT_VIDEO_PATH,fourcc, 30, (int(capture.get(3)),int(capture.get(4))))

	if not capture.isOpened:
	    print('ERROR: Unable to open video path ' + VIDEO_PATH)
	    exit(0)

	# cv.namedWindow("AveragedAxis", cv.WINDOW_NORMAL)
	cv.namedWindow("Final", cv.WINDOW_NORMAL)
	# cv.namedWindow("Edges", cv.WINDOW_NORMAL)
	x_axis_intersection = []
	angle = []
	length = []

	count = 0
	prev_line = None
	
	# Until frames keep coming
	while True:
	    _, frame = capture.read()
	    if frame is None:
	        break

	    fgMask = backSub.apply(frame)
	    fgMask1 = backSub1.apply(frame)

	    combined_fgMask = cv.bitwise_and(fgMask,fgMask1)

	    edges = cv.Canny(combined_fgMask, 50, 100, apertureSize=3)

	    denoised_edges = edges

	    middle_axis = find_median_axis(denoised_edges)

	    y_split = approximate_split(middle_axis)

	    lines = cv.HoughLines(middle_axis, 1, np.pi/180, 40)

	    if lines is not None:
	        pt1, pt2 = get_line(lines[0],y_split)
	        prev_line = (pt1, pt2)
	        cv.line(frame, pt1, pt2, (0, 0, 255), 6)
	    else:
	    	if prev_line is not None:
		        cv.line(frame, prev_line[0], prev_line[1], (0, 0, 255), 6)

	    out.write(frame)
	    cv.imshow('Final', frame)
	    # cv.imshow('AveragedAxis', middle_axis)
	    # cv.imshow('Edges', edges)

	    keyboard = cv.waitKey(30)
	    if keyboard == 'q' or keyboard == 27:
	        print('LOG: Exit from keyboard')
	        break
	    count += 1

	out.release()
	capture.release()
	cv.destroyAllWindows()