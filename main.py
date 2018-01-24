import os
import cv2
from parse_args import args
from cvinput import cvwindows
import detect_home

camera = cvwindows.create('camera')
i = 0
folder_path = os.listdir("videos")
folder_path.sort()
path = 'videos/' + folder_path[args.test_folder] + '/'
while cvwindows.event_loop():
    img_path = path + str(i) + '.png'
    frame = cv2.imread(img_path, 0)
    if frame is None:
        break
    cnt = detect_home.get_home(frame)
    if cnt is None:
        print i

    contours_img = frame.copy()
    cv2.drawContours(contours_img, [cnt], -1, (0, 255, 0), 2)
    camera.show(contours_img)
    i += 1
