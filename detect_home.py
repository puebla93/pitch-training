import argparse
import cv2
import numpy as np

args = None

def parse_args():
    parser = argparse.ArgumentParser(description="Pitvher Training")
    parser.add_argument('-c', "--camera", dest="camera",type=int, default=0, help='Index of the camera to use. Default 0, usually this is the camera on the laptop display')
    parser.add_argument('-d', "--debug", dest="debugging",type=bool, default=False, help='Print all windows. This option is gor debugging')

    return parser.parse_args()

def get_home(frame):
    # blur = cv2.GaussianBlur(frame, (11, 11), 0)
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(frame, 5)
    # _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)

    if args.debugging:
        cv2.imshow('Thresh', thresh)

    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if args.debugging:
        show_contours(contours, frame, 'all contours')

    contours = filter_by_area(frame, contours)
    contours = filter_by_sides(frame, contours)

    if __name__ == "__main__":
        # show_contours(contours, frame, 'Home Plate')
        contours_img = frame.copy()
        cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 2)
        cv2.imshow('Home Plate', contours_img)
        cv2.imshow('Original', frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours[0] if len(contours) == 1 else None

def filter_by_area(frame, contours):
    filter_contours = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        porcent_area = 100 * cnt_area / frame.size
        if porcent_area > 10 or porcent_area < 1:
            print "discarted by area"
        else:
            filter_contours.append(cnt)
            print "carted by area"
        if args.debugging:
            show_contours([cnt], frame, 'working cnt in filter by area')
            cv2.waitKey(0)
    if args.debugging:
        cv2.destroyWindow('working cnt in filter by area')
        cv2.destroyWindow('all contours')
        show_contours(filter_contours, frame, 'filter contours by area')
    return filter_contours

def filter_by_sides(frame, contours):
    filter_contours = []
    for cnt in contours:
        if args.debugging:
            show_contours([cnt], frame, 'working cnt in filter by sides')

        # hull = cv2.convexHull(cnt)
        # im = np.zeros((240, 320, 3), np.uint8)
        # cv2.drawContours(im, [hull], -1, (0, 255, 0), 1)
        # cv2.imshow("img", im)

        epsilon = 0.04*cv2.arcLength(cnt, True)
        cnt_approx = cv2.approxPolyDP(cnt, epsilon, True)
        if args.debugging:
            show_contours([cnt_approx], frame, 'approx of working cnt in filter by sides')

        if len(cnt_approx) != 5:
            print "discarted by sides"
            continue

        lines = get_lines_sorted_by_dist(cnt_approx)


        dist = get_dist(lines)
        min_diff = abs(dist[0] - dist[1])
        percent_min_sides = 100 * min_diff / dist[1] # find the percent with max distance to minimize the percentage
        max_diff = abs(dist[2] - dist[3])
        percent_middel_sides = 100 * max_diff / dist[3] # find the percent with max distance to minimize the percentage

        if args.debugging:
            print "\nPERCENT"
            print "MIN_SIDES: ", percent_min_sides
            print "MIDDEL_SIDES: ", percent_middel_sides

        # if min_diff < 5 and max_diff < 5: # filter by distances difference
        if percent_min_sides <= 20 and percent_middel_sides <= 20: # filter by percent between distances
            filter_contours.append(cnt)
        if args.debugging:
            if percent_min_sides > 20:
                print "discarted by min(blue) sides: ", percent_min_sides
            if percent_middel_sides > 20:
                print "discarted by middel(green) sides: ", percent_middel_sides
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.line(preview, (lines[0][0][0], lines[0][0][1]), (lines[0][1][0], lines[0][1][1]), (255, 0, 0), 1)
            cv2.line(preview, (lines[1][0][0], lines[1][0][1]), (lines[1][1][0], lines[1][1][1]), (255, 0, 0), 1)
            cv2.line(preview, (lines[2][0][0], lines[2][0][1]), (lines[2][1][0], lines[2][1][1]), (0, 255, 0), 1)
            cv2.line(preview, (lines[3][0][0], lines[3][0][1]), (lines[3][1][0], lines[3][1][1]), (0, 255, 0), 1)
            cv2.line(preview, (lines[4][0][0], lines[4][0][1]), (lines[4][1][0], lines[4][1][1]), (0, 0, 255), 1)
            cv2.imshow('lines', preview)
            cv2.waitKey(0)
    if args.debugging:
        cv2.destroyWindow('working cnt in filter by sides')
        cv2.destroyWindow('approx of working cnt in filter by sides')
        cv2.destroyWindow('lines')
        cv2.destroyWindow('filter contours by area')
        show_contours(filter_contours, frame, 'filter contours by sides')
    return filter_contours

def get_lines_sorted_by_dist(points):
    lines = []
    for i in range(len(points) - 1):
        lines.append([points[i][0], points[i+1][0]])
    lines.append([points[len(points) - 1][0], points[0][0]])
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

if __name__ == "__main__":
    args = parse_args()
    frame = cv2.imread('videos/Tue Jul  4 13:28:45 2017/0.png', 0)
    get_home(frame)
