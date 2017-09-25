import cv2
import numpy as np
from cvinput import cvwindows
from detect_home import get_home
from transform import four_point_transform

camera = cvwindows.create('camera')
i = 0
while cvwindows.event_loop():
    path = 'videos/Tue Jul  4 13:28:45 2017/' + str(i) + '.png'
    frame = cv2.imread(path, 0)
    if frame is None:
        break
    cnt = get_home(frame)
    if cnt is None:
        print i

    # rows,cols = frame.shape
    # pts = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    # M = cv2.getPerspectiveTransform(cnt,pts)

    # dst = cv2.warpPerspective(frame,M,(300,300))

    contours_img = frame.copy()
    cv2.drawContours(contours_img, [cnt], -1, (0, 255, 0), 3)
    camera.show(contours_img)
    i += 1
