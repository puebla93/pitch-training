import os
import cv2
import numpy as np
from utils import Obj
from utils import show_contours, draw_home_lines, refining_corners, angle, get_dist

params = Obj(
    debugging=False,
    thresh_blockSize=31,
    max_percentArea=10,
    min_percentArea=1,
    numberOfSizes=5,
    useHull=True,
    percentSideRatio=20,
    diff_rectAngles=5,
    diff_maxAngles=5
)

def get_homes(frame):
    thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, params.thresh_blockSize, 7)

    if params.debugging:
        cv2.imshow('Thresh', thresh)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if params.debugging:
        show_contours(contours, frame, 'all contours')

    filters = [filter_by_area, filter_by_sidesNumber, filter_by_sidesRatio, filter_by_angles]
    homes = filter_by(frame, filters, contours)

    if params.debugging:
        cv2.destroyWindow('Thresh')
        cv2.destroyWindow('filters contours by angles')
        print "DETECTING HOMES DONE!!!\n"
        show_contours(homes.astype('int32'), frame, 'homes')
        cv2.waitKey(0)
        cv2.destroyWindow('homes')

    return homes

def filter_by(frame, filters, contours):
    homes = contours
    for f in filters:
        homes = f(frame, homes)
    return homes

def filter_by_area(frame, contours):
    filter_contours = []
    frame_size = frame.shape[0] * frame.shape[1]
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        percent_area = 100 * cnt_area / frame_size
        
        if percent_area > params.max_percentArea or percent_area < params.min_percentArea:
            if params.debugging:
                print "discarded by area"
        else:
            filter_contours.append(cnt)
            if params.debugging:
                print "carded by area"
        
        if params.debugging:
            show_contours([cnt], frame, 'working cnt in filter by area')
            cv2.waitKey(0)

    if params.debugging:
        print '\nFilter By Area. DONE!!!\n'
        cv2.destroyWindow('working cnt in filter by area')
        cv2.destroyWindow('all contours')
        show_contours(filter_contours, frame, 'filters contours by area')
        cv2.waitKey(0)

    return np.array(filter_contours)

def filter_by_sidesNumber(frame, contours):
    filter_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)

        if params.useHull:
            epsilon = 0.03*cv2.arcLength(hull, True)
            cnt_approx = cv2.approxPolyDP(hull, epsilon, True)
        else:
            epsilon = 0.03*cv2.arcLength(cnt, True)
            cnt_approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(cnt_approx) == params.numberOfSizes:
            filter_contours.append(cnt_approx)
            if params.debugging:
                print "carded by sides number"
        elif params.debugging:
            print "discarded by sides number: ", len(cnt_approx), " sides"

        if params.debugging:
            show_contours([cnt], frame, 'working cnt in filter by sides')
            show_contours([hull], frame, "Hull")
            show_contours([cnt_approx], frame, 'approx of working cnt in filter by sides')
            cv2.waitKey(0)

    if params.debugging:
        print '\nFilter By Sides Number. DONE!!!\n'
        cv2.destroyWindow('working cnt in filter by sides')
        cv2.destroyWindow('Hull')
        cv2.destroyWindow('approx of working cnt in filter by sides')
        cv2.destroyWindow('filters contours by area')
        show_contours(filter_contours, frame, 'filters contours by sides number')
        cv2.waitKey(0)

    return np.array(filter_contours)

def filter_by_sidesRatio(frame, contours):
    filter_contours = []
    for cnt in contours:
        lines = get_lines_sorted_by_dist(cnt)

        min_dist = int(((lines[0][0][0] - lines[0][1][0])**2+(lines[0][0][1] - lines[0][1][1])**2)**.5)

        winSize = min_dist/2 if min_dist % 4 != 0 else min_dist/2 - 1
        refined_corners = cnt.astype('float32')
        refining_corners(frame, refined_corners, (winSize, winSize))
        refined_lines = get_lines_sorted_by_dist(refined_corners)

        dist = get_dist(refined_lines)

        min_diff = abs(dist[0] - dist[1])
        percent_min_sides = 100 * min_diff / dist[1] # find the percent with max distance to minimize the percentage
        max_diff = abs(dist[2] - dist[3])
        percent_middel_sides = 100 * max_diff / dist[3] # find the percent with max distance to minimize the percentage

        if percent_min_sides <= params.percentSideRatio and percent_middel_sides <= params.percentSideRatio: # filter by percent between distances
            filter_contours.append(refined_corners)
            if params.debugging:
                print 'carded by sides ratio'
        elif params.debugging:
            print "discarded by sides ratio"
            print "SORTED DISTANCES"
            print dist
            print "MIN_SIDES PERCENT: ", percent_min_sides
            print "MIDDEL_SIDES PERCENT: ", percent_middel_sides

        if params.debugging:
            show_contours([cnt], frame, 'working cnt in filter by sides ratio')
            draw_home_lines(lines, frame, 'lines')
            draw_home_lines(refined_lines, frame, 'refined_corners')
            cv2.waitKey(0)

    filter_contours = np.array(filter_contours)
    if params.debugging:
        print '\nFilter By Sides Ratio. DONE!!!\n'        
        cv2.destroyWindow('working cnt in filter by sides ratio')
        cv2.destroyWindow('lines')
        cv2.destroyWindow('refined_corners')
        cv2.destroyWindow('filters contours by sides number')
        show_contours(filter_contours.astype('int32'), frame, 'filters contours by sides ratio')
        cv2.waitKey(0)

    return filter_contours

def filter_by_angles(frame, contours):
    filter_contours = []
    for cnt in contours:
        angles = get_angles_sorted_by_magnitude(cnt)

        if np.allclose(angles[:3], [90]*3, 0, params.diff_rectAngles) and np.allclose(angles[3:], [135]*2, 0, params.diff_maxAngles):
            filter_contours.append(cnt)
            if params.debugging:
                print "carded by angles"
        elif params.debugging:
            print "discarded by angles"
            print "SORTED ANGLES"
            print angles

        if params.debugging:
            show_contours([cnt.astype('int32')], frame, 'working cnt in filter by angles')

    filter_contours = np.array(filter_contours)
    if params.debugging:
        print '\nFilter By Angles. DONE!!!\n'        
        cv2.destroyWindow('working cnt in filter by angles')
        cv2.destroyWindow('filters contours by sides ratio')
        show_contours(filter_contours.astype('int32'), frame, 'filters contours by angles')
        cv2.waitKey(0)

    return filter_contours

def get_lines_sorted_by_dist(cntPoints):
    lines = []
    for i in range(-1, len(cntPoints) - 1):
        lines.append([cntPoints[i][0], cntPoints[i+1][0]])
    lines.sort(key = (lambda line : ((line[0][0] - line[1][0])**2+(line[0][1] - line[1][1])**2)**.5))
    return lines

def get_angles_sorted_by_magnitude(cntPoints):
    angles = []
    pt0, pt1, pt2, pt3, pt4 = [pt[0] for pt in cntPoints]
    angles.append(angle(pt4-pt0, pt1-pt0))
    angles.append(angle(pt0-pt1, pt2-pt1))
    angles.append(angle(pt1-pt2, pt3-pt2))
    angles.append(angle(pt2-pt3, pt4-pt3))
    angles.append(angle(pt3-pt4, pt0-pt4))
    angles.sort()
    return angles

def setUp(nparams):
    params.setattr(nparams)
