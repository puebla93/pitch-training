import cv2
import numpy as np

def get_home(frame):
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_by_area(frame.size, contours)
    contours = filter_by_sides(contours)

    # contours_img = frame.copy()
    # cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 3)
    # cv2.imshow('Home Plate', contours_img)
    # cv2.imshow('Original', thresh)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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
        # cv2.drawContours(im, [hull], -1, (0, 255, 0), 3)
        # cv2.imshow("img", im)

        # epsilon = 0.05*cv2.arcLength(cnt, True)
        epsilon = 0.05*cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        lines = get_lines(approx)

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
        if abs(dist[0] - dist[1]) < 4 and abs(dist[2] - dist[3]) < 3:
            # filter_contours.append(cnt)
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
    # for d in dist:
    #     print d
    dist.sort()
    return dist

if __name__ == "__main__":
    frame = cv2.imread('videos/Tue Jul  4 13:28:01 2017/457.png', 0)
    get_home(frame)
