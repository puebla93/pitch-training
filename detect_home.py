import cv2
import numpy as np
from cvinput import cvwindows

def get_home():
    frame = cv2.imread('videos/Tue Jul  4 13:28:01 2017/462.png', 0)
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_by_area(frame.size, contours)
    contours = filter_by_sides(contours)

    contours_img = frame.copy()
    cv2.drawContours(contours_img, contours, -1, (0, 0, 255), 3)
    cv2.imshow('Home Plate', contours_img)
    cv2.imshow('Original', thresh)

    while cvwindows.event_loop():
        pass
    return contours[0] if len(contours) == 1 else None

def filter_by_area(img_area, contours):
    filter_contours = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        porcent_area = 100 * cnt_area / img_area
        if porcent_area < 10:
            filter_contours.append(cnt)
    return filter_contours

def filter_by_sides(contours):
    filter_contours = []
    for cnt in contours:
        epsilon = 0.05*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        lines = get_lines(approx)
        dist = get_sort_dist(lines)
        if abs(dist[0] - dist[1]) < 1 and abs(dist[2] - dist[3]):
            filter_contours.append(cnt)
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
    get_home()
