"""
All functions combined
"""
import math
import pickle
from scipy.spatial import distance as dist

import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_game(frame, size, text):
    """
    Draw game using game_object on frame.
    """
    # pos_float = game_obj['pos']
    # pos_int = ((int)(pos_float[0]), (int)(pos_float[1]))
    # frame = cv2.circle(frame, pos_int, 15, (0, 0, 255), -1)
    # frame = cv2.rectangle(frame, (15, 15), (640-15, 480-15), (0, 255, 0), 1)

    size_x = size[1]
    size_y = size[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left = (10, size_y-10)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(frame, str(text),
                bottom_left,
                font,
                font_scale,
                font_color,
                line_type)

    return frame

