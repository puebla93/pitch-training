import os
import numpy as np
import cv2
from cvinput import cvwindows
from kmeans import kmeans
from parse_args import args
from params import params
from utils import show_contours, draw_circle, draw_circles

def get_ball(fgbg, frame, kernel=None):
    fgmask = fgbg.apply(frame)
    if kernel is not None:
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    if args.debugging:
        cv2.imshow('BackgroundSubtractorMOG2', fgmask)

    if params.useKmeans:
        fgmask = kmeans(fgmask, params.KmeansK)
        if args.debugging:
            cv2.imshow('Kmeans', fgmask)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if args.debugging:
        show_contours(contours, frame, 'all contours')

    centers, radiuses = filter_by_radius(frame, contours)

    return centers, radiuses

def filter_by_radius(frame, contours):
    centers = []
    radiuses = []
    for cnt in contours:
        if args.debugging:
            show_contours([cnt], frame, 'working cnt in filter by radius')

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        if radius > params.max_percentRadius or radius < params.min_percentRadius:
            if args.debugging:
                print "discarded by radius"
        else:
            centers.append(center)
            radiuses.append(radius)
            if args.debugging:
                print "carded by radius"

        if args.debugging:
            draw_circle(center, radius, frame, 'Enclosing circle of working cnt in filter by radius')
            # cv2.waitKey(0)

    if args.debugging:
        draw_circles(centers, radiuses, frame, 'filters circles by radius')

    return centers, radiuses

if __name__ == '__main__':
    camera = cvwindows.create('balls')

    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/'

    mog2 = cv2.BackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    i = 0
    ball_tracking = []
    while cvwindows.event_loop():
        img_path = path + str(i) + '.png'
        frame = cv2.imread(img_path)
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centers, radiuses = get_ball(mog2, gray, kernel)
        ball_tracking.append((centers, radiuses))
        for j in range(len(centers)):
            cv2.circle(frame, centers[j], radiuses[j], (0, 255, 0), 2)
        camera.show(frame)
        if args.debugging:
            cv2.waitKey(0)

        i += 1

    cv2.destroyAllWindows()

    img_path = path + '0.png'
    frame = cv2.imread(img_path)
    for centers, radiuses in ball_tracking:
        for j in range(len(centers)):
            cv2.circle(frame, centers[j], radiuses[j], (0, 255, 0), 2)
    cv2.imshow("tracking", frame)
    cv2.waitKey()
