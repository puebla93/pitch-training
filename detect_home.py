import cv2
import numpy as np
from cvinput import cvwindows

def main():
    frame = cv2.imread('0.png', 0)
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours_img = thresh.copy()
    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = np.zeros((240, 320, 3), np.uint8)
    cv2.drawContours(contours_img, contours, -1, (255, 0, 255), 3)

    contours = filter_by_area(frame.size, contours)

    contours = filter_by_sides(contours)

    cv2.imshow('Thresh', thresh)
    # cv2.imshow('Opening', opening)
    cv2.imshow('Contours_img', contours_img)

    while cvwindows.event_loop():
        pass

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

        points = max_dist(approx)
        # fd
        approx = np.zeros((240, 320, 3), np.uint8)
        cv2.drawContours(approx, contours, -1, (255, 0, 0), 3)
        cv2.drawContours(approx, points, -1, (0, 0, 255), 3)
        cv2.imshow("a", approx)

    return filter_contours

def max_dist(points):
    result = []
    _max = 0
    for i in range(len(points) - 1):
        dist = ((points[i][0][0]-points[i+1][0][0])**2 + (points[i][0][1]-points[i+1][0][1])**2)**(1./2)
        if dist > _max:
            _max = dist
            result.append(points[i])
            result.append(points[i+1])
    dist = ((points[0][0][0]-points[len(points)-1][0][0])**2 + (points[0][0][1]-points[len(points)-1][0][1])**2)**(1./2)
    if dist > _max:
        _max = dist
        result.append(points[i])
        result.append(points[i+1])
    return result

def get_lines(points):
    lines = []
    for i in range(len(points) - 1):
        lines.append([points[i][0], points[i+1][0]])
    lines.append([points[len(points) - 1][0], points[0][0]])
    return lines

if __name__ == "__main__":
    main()
