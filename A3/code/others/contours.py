import cv2

if __name__ == "__main__":

    VID_FEED = cv2.VideoCapture(-1)

    while True:
        RET, FRAME = VID_FEED.read()
        if not RET:
            print("Unable to capture video")
            sys.exit()

        gray = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        contours, hierarchy = cv2.findContours(
            edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        conv_cont = []
        for cnt in contours:
            if cv2.isContourConvex(cnt):
                conv_cont.append(cnt)
                print('True')
        cv2.drawContours(FRAME, conv_cont, -1, (0, 255, 0), 3)
        
        
        cv2.imshow('frame', FRAME)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    VID_FEED.release()
    cv2.destroyAllWindows()