import os
import cv2
import numpy as np
from refine_corner import refining_corners
from parse_args import args
from params import params
from kmeans import kmeans

def get_home(frame):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(frame, params.medianBlur_ksize)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, params.thresh_blockSize, 7)

    if args.debugging:
        cv2.imshow('Blur', blur)
        cv2.imshow('Thresh', thresh)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if args.debugging:
        show_contours(contours, frame, 'all contours')

    contours = filter_by_area(frame, contours)
    contours = filter_by_sidesNumber(frame, contours)
    contours = filter_by_sidesRatio(frame, contours)

    if args.debugging:
        cv2.destroyWindow('Blur')
        cv2.destroyWindow('Thresh')
        cv2.destroyWindow('filters contours by sides ratio')

    if __name__ == "__main__":
        show_contours(contours.astype('int32'), frame, 'Home Plate')
        cv2.imshow('Original', frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contours
    # return contours[0] if len(contours) == 1 else None

def filter_by_area(frame, contours):
    filter_contours = []
    frame_size = frame.shape[0] * frame.shape[1]
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        percent_area = 100 * cnt_area / frame_size
        if percent_area > params.max_percentArea or percent_area < params.min_percentArea:
            if args.debugging:
                print "discarded by area"
                print "area = ", cnt_area, " percent = ", percent_area
        else:
            filter_contours.append(cnt)
            if args.debugging:
                print "carded by area"
        if args.debugging:
            show_contours([cnt], frame, 'working cnt in filter by area')
            cv2.waitKey(0)
    if args.debugging:
        cv2.destroyWindow('working cnt in filter by area')
        cv2.destroyWindow('all contours')
        show_contours(filter_contours, frame, 'filters contours by area')
        cv2.waitKey(0)
    return np.array(filter_contours)

def filter_by_sidesNumber(frame, contours):
    filter_contours = []
    for cnt in contours:
        ###############################################

        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # show_contours([box], frame, 'Rect')
        # im = cv2.drawContours(im,[box],0,(0,0,255),2)

        ###############################################

        hull = cv2.convexHull(cnt)

        if args.debugging:
            show_contours([cnt], frame, 'working cnt in filter by sides')
            show_contours([hull], frame, "Hull")

        if params.useHull:
            epsilon = 0.03*cv2.arcLength(hull, True)
            cnt_approx = cv2.approxPolyDP(hull, epsilon, True)
        else:
            epsilon = 0.03*cv2.arcLength(cnt, True)
            cnt_approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(cnt_approx) != params.numberOfSizes:
            if args.debugging:
                print "discarded by sides"
                print len(cnt_approx), " sides"
        else:
            filter_contours.append(cnt_approx)

        if args.debugging:
            show_contours([cnt_approx], frame, 'approx of working cnt in filter by sides')
            cv2.waitKey(0)

    if args.debugging:
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

        if args.debugging:
            print "\nPERCENT"
            print "MIN_SIDES: ", percent_min_sides
            print "MIDDEL_SIDES: ", percent_middel_sides

        if percent_min_sides <= params.percentSideRatio and percent_middel_sides <= params.percentSideRatio: # filter by percent between distances
            filter_contours.append(refined_corners)

        if args.debugging:
            if percent_min_sides > params.percentSideRatio:
                print "discarded by min(blue) sides: ", percent_min_sides
            if percent_middel_sides > params.percentSideRatio:
                print "discarded by middel(green) sides: ", percent_middel_sides
            draw_lines(lines, frame, 'lines')
            draw_lines(refined_lines, frame, 'refined_corners')
            cv2.waitKey(0)

    filter_contours = np.array(filter_contours)
    if args.debugging:
        cv2.destroyWindow('lines')
        cv2.destroyWindow('refined_corners')
        cv2.destroyWindow('filters contours by sides number')
        show_contours(filter_contours.astype('int32'), frame, 'filters contours by sides ratio')
        cv2.waitKey(0)

    return filter_contours

def get_lines_sorted_by_dist(points):
    lines = []
    for i in range(-1, len(points) - 1):
        lines.append([points[i][0], points[i+1][0]])
    lines.sort(key = (lambda line : ((line[0][0] - line[1][0])**2+(line[0][1] - line[1][1])**2)**.5))
    return lines

def get_dist(lines):
    dist = []
    for line in lines:
        dist.append(((line[0][0] - line[1][0])**2+(line[0][1] - line[1][1])**2)**.5)
    if args.debugging:
        print "\nDISTANCES SORTED"
        print dist
    return dist

def show_contours(cnt, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, cnt, -1, (0, 0, 255), 1)
    cv2.imshow(window_name, preview)

def draw_lines(lines, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.line(preview, (lines[0][0][0], lines[0][0][1]), (lines[0][1][0], lines[0][1][1]), (255, 0, 0), 1)
    cv2.line(preview, (lines[1][0][0], lines[1][0][1]), (lines[1][1][0], lines[1][1][1]), (255, 0, 0), 1)
    cv2.line(preview, (lines[2][0][0], lines[2][0][1]), (lines[2][1][0], lines[2][1][1]), (0, 255, 0), 1)
    cv2.line(preview, (lines[3][0][0], lines[3][0][1]), (lines[3][1][0], lines[3][1][1]), (0, 255, 0), 1)
    cv2.line(preview, (lines[4][0][0], lines[4][0][1]), (lines[4][1][0], lines[4][1][1]), (0, 0, 255), 1)
    cv2.imshow(window_name, preview)

if __name__ == "__main__":
    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/' + str(args.test_frame) + '.png'
    frame = cv2.imread(path, 0)
    if params.useKmeans:
        frame = kmeans(frame, params.kmeans_k)
    get_home(frame)
