import os
import numpy as np
import cv2
from utils import Obj, Ball
from utils import show_contours, draw_ball, draw_balls, kmeans

params = Obj(
    debugging=False,
    useKmeans=False,
    kmeans_k=6,
    max_radiusPercent=.01,
    min_radiusPercent=.0015,
    fgbg=cv2.BackgroundSubtractorMOG2(),
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    aproxContour=0 # 0 It is a straight rectangle, it doesn’t consider the rotation of the contour
                   # 1 Drawn with minimum area, so it considers the rotation also.
                   # 2 It is a circle which completely covers the contour with minimum area
)

def get_balls(frame):
    mask = get_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if params.debugging:
        show_contours(contours, frame, 'all contours')

    filters = [filter_by_radius]
    balls = filter_by(frame, filters, contours)

    if params.debugging:
        cv2.destroyWindow('filters balls by radius')
        if params.useKmeans:
            cv2.destroyWindow('Kmeans')
        else:
            cv2.destroyWindow('Background Subtractor')
        print "\nGETTING BALLS DONE!!!\n"        
        draw_balls(balls, frame, 'balls')
        cv2.waitKey(0)
        cv2.destroyWindow('balls')

    return balls

def get_mask(frame):
    if params.useKmeans:
        fgmask = kmeans(frame, params.KmeansK)
        if params.debugging:
            cv2.imshow('Kmeans', fgmask)
    else:
        fgmask = params.fgbg.apply(frame)
        if params.kernel is not None:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, params.kernel)
        if params.debugging:
            cv2.imshow('Background Subtractor', fgmask)
    # cv2.waitKey(0)

    return fgmask

def filter_by(frame, filters, contours):
    balls = contours
    for f in filters:
        balls = f(frame, balls)
    return balls

def filter_by_radius(frame, contours):
    balls = []
    frame_size = frame.shape[0] * frame.shape[1]    
    for cnt in contours:
        if params.aproxContour == 0:
            x,y,w,h = cv2.boundingRect(cnt)
            center = np.array([x+w/2., y+h/2.])
            radius = w/2. if w < h else h/2.
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        elif params.aproxContour == 1:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

        elif params.aproxContour == 2:
            center, radius = cv2.minEnclosingCircle(cnt)

        ball = Ball(center, radius)       
        radiusPercent = 100 * radius / frame_size

        if radiusPercent > params.max_radiusPercent or radiusPercent < params.min_radiusPercent:
            if params.debugging:
                print "discarded by radius ", radiusPercent        
        else:
            balls.append(ball)
            if params.debugging:
                print "carded by radius"

        if params.debugging:
            show_contours([cnt], frame, 'working cnt in filter by radius')
            draw_ball(ball, frame, 'Enclosing circle of working cnt in filter by radius')
            cv2.waitKey(0)

    if params.debugging:
        cv2.destroyWindow('working cnt in filter by radius')
        cv2.destroyWindow('Enclosing circle of working cnt in filter by radius')
        cv2.destroyWindow('all contours')
        draw_balls(balls, frame, 'filters balls by radius')
        cv2.waitKey(0)

    return np.array(balls)

def setUp(nparams):
    params.setattr(nparams)
