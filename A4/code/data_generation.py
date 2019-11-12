import copy
import sys
import cv2
import time
import numpy as np

if __name__ == "__main__":
    VID_FEED = cv2.VideoCapture(sys.argv[2])
    i = 0
    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        FRAME = np.rot90(FRAME, 3)
        cv2.imshow("frame", FRAME)
        # if i%10 == 0:
        # cv2.imwrite(sys.argv[1] + str(time.time())+".jpg", FRAME)
        i += 1
    
    VID_FEED.release()
    cv2.destroyAllWindows()
