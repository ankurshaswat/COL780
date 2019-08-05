from os import listdir, mkdir
from os.path import isfile, join , exists
import cv2 

# Function to extract frames


def FrameCapture(video_name, path):
    if not exists("../frames"):
        mkdir("../frames")

    if not exists("../frames/"+video_name):
        mkdir("../frames/"+video_name)

    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while True:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # Saves the frames with frame-count
        if success:
            cv2.imwrite("../frames/"+video_name+"/frame%d.jpg" % count, image)
        else:
            break
            
        count += 1


mypath = '../videos'

videos = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for video in videos:
    video_name = video.split('.')[0]
    video_path = mypath + '/' + video
    print(video_name,video_path)
    FrameCapture(video_name, video_path)
