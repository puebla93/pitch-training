import argparse
import cv2
from cvinput import cvwindows
import detect_home


def parse_args():
    parser = argparse.ArgumentParser(description="Pitvher Training")
    parser.add_argument('-c', "--camera", dest="camera",type=int, default=0, help='Index of the camera to use. Default 0, usually this is the camera on the laptop display')
    parser.add_argument('-d', "--debug", dest="debugging",type=bool, default=False, help='Print all windows. This option is gor debugging')

    return parser.parse_args()


detect_home.args = parse_args()
camera = cvwindows.create('camera')
i = 0
while cvwindows.event_loop():
    path = 'videos/Tue Jul  4 13:28:45 2017/' + str(i) + '.png'
    frame = cv2.imread(path, 0)
    if frame is None:
        break
    cnt = detect_home.get_home(frame)
    if cnt is None:
        print i

    contours_img = frame.copy()
    cv2.drawContours(contours_img, [cnt], -1, (0, 255, 0), 3)
    camera.show(contours_img)
    i += 1
