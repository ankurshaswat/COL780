import math
import copy

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from tqdm import tqdm

ALGO = 'KNN'  # MOG2 or KNN
VIDEO_PATH = '../videos/1.mp4'
OUT_VIDEO_PATH = '../out_videos/1.avi'
NUMBER_LINES_CUTOFF = 5
AXIS_SIZE = 3
LENGTH_PERCENTAGE_THRESH = 70

def find_median_axis(frame):
    indices = np.argwhere(edges > 0)

    averaged_median_axis = copy.deepcopy(edges)

    for i in range(edges.shape[0]):
        if len(indices) > 0:
            averaged_median_axis[i, :] = 0
            pixel = np.average(indices[indices[:, 0] == i], axis=0)[1]
            if not math.isnan(pixel):
                pixel = np.int16(pixel).item()
                for x in range(max(0, pixel-AXIS_SIZE), min(np.int16(averaged_median_axis.shape[1]).item(), pixel+AXIS_SIZE)):
                    averaged_median_axis[i][x] = 255

    return averaged_median_axis


def get_line(lines,y2):
    avg = [0, 0, 0, 0]

    num = min(len(lines), NUMBER_LINES_CUTOFF)

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


def get_intersection(lines):
    num = min(len(lines), NUMBER_LINES_CUTOFF)
    fin_x = 0.0
    angle = 0.0
    for rho, theta in lines[0:min(len(lines), NUMBER_LINES_CUTOFF)]:
        a = np.cos(theta)
        b = np.sin(theta)
        angle += (a/b)
        x0 = a*rho
        y0 = b*rho
        fin_x += x0 + (b*y0 / a)
    return fin_x / num, angle / num

def denoise(frame):
    return cv.fastNlMeansDenoising(frame, None, 30.0, 7, 21)

def approximate_split(frame):
    frame_copy = copy.deepcopy(frame)
    indices = (frame_copy>0)*np.arange(frame_copy.shape[1])
    # variances = np.var(indices,axis=1)
    sum_ = np.sum(frame_copy>0,axis=1)
    cum_sum = np.cumsum(sum_)
    percentages = 100*cum_sum/cum_sum[-1]

    # weights = [0.1,0.2,0.4,0.2,0.1]
    # cum_sum_smooth = np.convolve(cum_sum,np.array(weights)[::-1],'same')

    # derivative = []
    # for i in range(2,len(cum_sum)-2):
    #     derivative.append(cum_sum_smooth[i]-cum_sum_smooth[i-1])
    # plt.plot(cum_sum_smooth)
    # plt.show()

    for i in range(percentages.shape[0]):
        if(percentages[i]>LENGTH_PERCENTAGE_THRESH):
            return i

    return 0

if ALGO == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
    # backSub.setHistory(100)

capture = cv.VideoCapture(cv.samples.findFileOrKeep(VIDEO_PATH))

if not capture.isOpened:
    print('ERROR: Unable to open video path ' + VIDEO_PATH)
    exit(0)

cv.namedWindow("Frame", cv.WINDOW_NORMAL)
x_axis_intersection = []
angle = []
length = []

frame_size = 0
# print((frame.shape[0],frame.shape[1]))
out = None
while True:
    _, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    edges = cv.Canny(fgMask, 50, 100, apertureSize=3)

    # denoised_edges = denoise(edges)
    denoised_edges = edges

    middle_axis = find_median_axis(denoised_edges)

    y_split = approximate_split(denoised_edges)
    length.append(y_split)

    lines = cv.HoughLines(middle_axis, 1, np.pi/180, 40)

    if lines is not None:
        pt1, pt2 = get_line(lines[0],y_split)
        data_x,data_angle = get_intersection(lines[0])

        x_axis_intersection.append(data_x)
        angle.append(data_angle)
        cv.line(frame, pt1, pt2, (0, 0, 255), 6)
    else:
        x_axis_intersection.append(0)
        angle.append(0)

    if frame_size == 0:
        frame_size = frame.shape
        print((int(capture.get(3)),int(capture.get(4))))
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(OUT_VIDEO_PATH,fourcc, 30, (int(capture.get(3)),int(capture.get(4))))

    cv.imshow('Frame', frame)
    out.write(frame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        print('LOG: Exit from keyboard')
        break

out.release()
r = np.arange(len(x_axis_intersection))
weights = [0.1,0.2,0.4,0.2,0.1]
x_axis_intersection_smooth = np.convolve(x_axis_intersection,np.array(weights)[::-1],'same')
angle_smooth = np.convolve(angle,np.array(weights)[::-1],'same')
length_smooth = np.convolve(length,np.array(weights)[::-1],'same')

plt.figure(1)
plt.ylabel('X axis intersection')
plt.plot(r,x_axis_intersection)
plt.figure(2)
plt.ylabel('Line Angle')
plt.plot(r,angle)
plt.figure(3)
plt.ylabel('Length')
plt.plot(r,length)
# plt.show()


plt.figure(4)
plt.ylabel('X axis intersection S')
plt.plot(r,x_axis_intersection_smooth)
plt.figure(5)
plt.ylabel('Line Angle S')
plt.plot(r,angle_smooth)
plt.figure(6)
plt.ylabel('Length S')
plt.plot(r,length_smooth)
# plt.show()
## Observe Plots
## We can do running average to smoothen these curves
## Try any other idea for length