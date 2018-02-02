import os
import numpy as np
import cv2
from kmeans import kmeans
from utils import show_contours, draw_circle, draw_circles, Obj

params = Obj(
    debugging=False,
    useKmeans=False,
    kmeans_k=6,
    max_percentRadius=10,
    min_percentRadius=1.5
)

def get_ball(frame, fgbg=None, kernel=None):
    mask = get_mask(frame, fgbg, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if params.debugging:
        show_contours(contours, frame, 'all contours')

    centers, radiuses = filter_by_radius(frame, contours)

    return centers, radiuses

def get_mask(frame, fgbg, kernel):
    if params.useKmeans:
        fgmask = kmeans(frame, params.KmeansK)
        if params.debugging:
            cv2.imshow('Kmeans', fgmask)
    else:
        fgmask = fgbg.apply(frame)
        if kernel is not None:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        if params.debugging:
            cv2.imshow('BackgroundSubtractorMOG2', fgmask)
    return fgmask

def filter_by_radius(frame, contours):
    centers = []
    radiuses = []
    for cnt in contours:
        if params.debugging:
            show_contours([cnt], frame, 'working cnt in filter by radius')

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = x, y
        radius = radius

        if radius > params.max_percentRadius or radius < params.min_percentRadius:
            if params.debugging:
                print "discarded by radius"
        else:
            centers.append(center)
            radiuses.append(radius)
            if params.debugging:
                print "carded by radius"

        if params.debugging:
            draw_circle(center, radius, frame, 'Enclosing circle of working cnt in filter by radius')
            cv2.waitKey(0)

    if params.debugging:
        draw_circles(centers, radiuses, frame, 'filters circles by radius')

    return centers, radiuses

def setUp(nparams):
    params.setattr(nparams)
