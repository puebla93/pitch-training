import os
import numpy as np
import cv2
from parse_args import args
from cvinput import cvwindows
from params import params
from kmeans import kmeans

def get_balls(fgbg, frame, kernel=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)
    if kernel is not None:
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    if args.debugging:
        cv2.imshow('BackgroundSubtractorMOG2', fgmask)

    if params.useKmeans:
        fgmask = kmeans(fgmask, params.KmeansK)
        cv2.imshow('Kmeans', fgmask)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if args.debugging:
        show_contours(contours, gray, 'all contours')

    centers = []
    radiuses = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)

        if args.debugging:
            show_contours([cnt], gray, 'working cnt in filter by sides')
            show_contours([hull], gray, "Hull")

        if params.useHull:
            (x, y), radius = cv2.minEnclosingCircle(hull)
        else:
            (x, y), radius = cv2.minEnclosingCircle(hull)

        center = (int(x), int(y))
        radius = int(radius)

        centers.append(center)
        radiuses.append(radius)

        if args.debugging:
            draw_circle(center, radius, gray, 'approx circle of working cnt')
            # cv2.waitKey(0)

    return centers, radiuses

def show_contours(cnt, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, cnt, -1, (0, 0, 255), 1)
    cv2.imshow(window_name, preview)

def draw_circle(center, radius, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.circle(preview, center, radius, (0, 255, 0), 2)
    cv2.imshow(window_name, preview)

if __name__ == '__main__':
    camera = cvwindows.create('balls')

    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/'

    mog2 = cv2.BackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # i = 442
    i = 0
    d = {}
    while cvwindows.event_loop():
        print i
        img_path = path + str(i) + '.png'
        frame = cv2.imread(img_path)
        if frame is None:
            break
        # if i > 463:
        #     break

        centers, radiuses = get_balls(mog2, frame, kernel)
        for j in range(len(centers)):
            # preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.circle(frame, centers[j], radiuses[j], (0, 255, 0), 2)
        camera.show(frame)
        if len(centers) != 0:
            cv2.waitKey(0)

        i += 1
