import cv2
import numpy as np

def get_home(frame):
    # blur = cv2.GaussianBlur(frame, (11, 11), 0)
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(frame, 5)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # im = np.zeros((240, 320, 3), np.uint8)
    # cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
    # cv2.imshow('a', im)

    contours = filter_by_area(frame.size, contours)
    contours = filter_by_sides(contours)

    if __name__ == "__main__":
        contours_img = frame.copy()
        cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 3)
        cv2.imshow('Home Plate', contours_img)
        cv2.imshow('Thresh', thresh)
        cv2.imshow('Original', frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours[0] if len(contours) == 1 else None

def filter_by_area(img_area, contours):
    filter_contours = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        porcent_area = 100 * cnt_area / img_area
        if porcent_area < 10 and porcent_area > 1:
            filter_contours.append(cnt)
    return filter_contours

def filter_by_sides(contours):
    filter_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        # im = np.zeros((240, 320, 3), np.uint8)
        # cv2.drawContours(im, [hull], -1, (0, 255, 0), 1)
        # cv2.imshow("img", im)

        epsilon = 0.05*cv2.arcLength(hull, True)
        cnt_approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(cnt_approx) != 5:
            continue

        lines = get_lines(cnt_approx)

        # img = np.zeros((240, 320, 3), np.uint8)
        # cv2.line(img, (lines[0][0][0], lines[0][0][1]), (lines[0][1][0], lines[0][1][1]), (255, 0, 0), 1)
        # cv2.line(img, (lines[1][0][0], lines[1][0][1]), (lines[1][1][0], lines[1][1][1]), (0, 255, 0), 1)
        # cv2.line(img, (lines[2][0][0], lines[2][0][1]), (lines[2][1][0], lines[2][1][1]), (0, 0, 255), 1)
        # cv2.line(img, (lines[3][0][0], lines[3][0][1]), (lines[3][1][0], lines[3][1][1]), (255, 255, 0), 1)
        # cv2.line(img, (lines[4][0][0], lines[4][0][1]), (lines[4][1][0], lines[4][1][1]), (255, 0, 255), 1)
        # cv2.line(img, (lines[5][0][0], lines[5][0][1]), (lines[5][1][0], lines[5][1][1]), (0, 255, 255), 1)
        # cv2.line(img, (lines[6][0][0], lines[6][0][1]), (lines[6][1][0], lines[6][1][1]), (255, 255, 255), 1)
        # cv2.imshow('a', img)

        dist = get_sort_dist(lines)
        min_diff = abs(dist[0] - dist[1])
        percent_min_sides = 100 * min_diff / dist[1] # find the percent with max distance to minimize the percentage
        max_diff = abs(dist[2] - dist[3])
        percent_midde_sides = 100 * max_diff / dist[3] # find the percent with max distance to minimize the percentage
        # if min_diff < 5 and max_diff < 5: # filter by distances difference
        if percent_min_sides < 15 and percent_midde_sides < 15: # filter by percent between distances
            filter_contours.append(hull)
    return filter_contours

def get_lines(points):
    lines = []
    for i in range(len(points) - 1):
        lines.append([points[i][0], points[i+1][0]])
    lines.append([points[len(points) - 1][0], points[0][0]])
    return lines

def get_sort_dist(lines):
    dist = []
    for line in lines:
        dist.append(((line[0][0] - line[1][0])**2+(line[0][1] - line[1][1])**2)**.5)
    dist.sort()
    return dist

if __name__ == "__main__":
    frame = cv2.imread('videos/Tue Jul  4 13:28:45 2017/0.png', 0)
    get_home(frame)
