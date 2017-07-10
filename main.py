import cv2
import numpy as np
from cvinput import cvwindows
from detect_home import get_home

camera = cvwindows.create('camera')
for i in range(727):
    path = 'videos/Tue Jul  4 13:26:23 2017/' + str(i) + '.png'
    frame = cv2.imread(path, 0)
    cnt = get_home(frame)
    if cnt is None:
        print i
    contours_img = frame.copy()
    cv2.drawContours(contours_img, [cnt], -1, (0, 255, 0), 3)
    if not cvwindows.event_loop():
        break
    camera.show(contours_img)

