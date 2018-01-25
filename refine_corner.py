import cv2

def refining_corners(gray, corners, winSize):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cv2.cornerSubPix(gray, corners, winSize, (-1, -1), criteria)
