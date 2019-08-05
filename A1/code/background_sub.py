import cv2 as cv

algo = 'MOG2' # MOG2 or KNN
frame_seq_path = '../videos/1.mp4'

if algo=='MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(frame_seq_path))
if not capture.isOpened:
    print('Unable to open: ' + frame_seq_path)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        print('Frames over')
        break
    
    fgMask = backSub.apply(frame)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        print('Exit from keyboard')
        break