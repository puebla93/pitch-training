import os
import numpy as np
import cv2
from utils import show_contours, draw_ball, draw_balls, Obj, kmeans

params = Obj(
    debugging=False,
    useKmeans=False,
    kmeans_k=6,
    max_radiusPercent=.01,
    min_radiusPercent=.002,
    fgbg=cv2.BackgroundSubtractorMOG2(),
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
)

def get_balls(frame):
    mask = get_mask(frame)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = np.array([x, y])
        radiusPercent = 100 * radius / frame_size

        if radiusPercent > params.max_radiusPercent or radiusPercent < params.min_radiusPercent:
            if params.debugging:
                print "discarded by radius"
        else:
            balls.append(np.array([center, radius]))
            if params.debugging:
                print "carded by radius"

        if params.debugging:
            show_contours([cnt], frame, 'working cnt in filter by radius')
            draw_ball((center, radius), frame, 'Enclosing circle of working cnt in filter by radius')
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
