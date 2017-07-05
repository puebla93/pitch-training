import cv2
import numpy as np
from cvinput import cvwindows

def main():
    frame = cv2.imread('0.png', 0)
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(opening, contours, -1, (255, 0, 0), 3)

    cv2.imshow('Thresh', thresh)
    cv2.imshow('Opening', opening)

    while cvwindows.event_loop():
        pass

if __name__ == "__main__":
    main()
