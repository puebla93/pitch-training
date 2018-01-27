import os
import cv2
from parse_args import args
from cvinput import cvwindows
import detect_home
from kmeans import kmeans
from params import params

camera = cvwindows.create('camera')
i = 0
folder_path = os.listdir("videos")
folder_path.sort()
path = 'videos/' + folder_path[args.test_folder] + '/'
while cvwindows.event_loop():
    img_path = path + str(i) + '.png'
    frame = cv2.imread(img_path)
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if params.useKmeans:
        gray = kmeans(gray, params.kmeans_k)
    cnt = detect_home.get_home(gray)
    if cnt is None or len(cnt) == 0:
        print i

    contours_img = frame.copy()
    cv2.drawContours(contours_img, cnt.astype('int32'), -1, (0, 0, 255), 2)
    camera.show(contours_img)
    i += 1
