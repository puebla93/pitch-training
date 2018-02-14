import math
import cv2
import numpy as np

class Obj(object):
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            self.__setattr__(k, v)
    def setattr(self, kwarg):
        for k, v in kwarg.items():
            self.__setattr__(k, v)
    def __repr__(self):
        return str(self.__dict__)
    def __str__(self):
        return str(self.__dict__)

class Reader():
    def __init__(self):
        self._actual_frame = -1
        self.__setDefaults__()
        self.capture = None
        if self.params.read_from == "camera":
            self.capture = cv2.VideoCapture(self.params.camera_index)

    def read(self):
        self._actual_frame += 1
        if self.params.read_from == "camera":
            return self.capture.read()[1]
        if self.params.read_from == "folder":
            path = self.params.folder_path + str(self._actual_frame) + self.params.frame_type
            frame = cv2.imread(path)
            return frame

    def restart_reading(self):
        if self.params.read_from == "camera":
            self.capture.release()
            self.capture = cv2.VideoCapture(self.params.camera_index)
        elif self.params.read_from == "folder":
            self._actual_frame = 0

    def setUp(self, nparams):
        self.params.setattr(nparams)
        if nparams.has_key("read_from") and nparams["read_from"] == "camera" and self.capture is None:
            self.capture = cv2.VideoCapture(self.params.camera_index)

    def get_actualFrame(self):
        return self._actual_frame

    def __setDefaults__(self):
        self.params = Obj(
            read_from="folder",
            folder_path="videos/Tue Jul  4 13:26:23 2017/",
            frame_type=".png",
            camera_index=0
            )

class HomePlate():
    def __init__(self, cnt):
        self.cnt = cnt
        pt0, pt1, pt2, pt3, pt4 = self.__split_cnt__(cnt)
        self.ordered_pts = self.__find_order__(pt0, pt1, pt2, pt3, pt4)

    def __find_order__(self, pt0, pt1, pt2, pt3, pt4):
        alpha0 = angle(np.abs(pt4-pt0), np.abs(pt1-pt0))
        alpha1 = angle(np.abs(pt0-pt1), np.abs(pt2-pt1))
        alpha2 = angle(np.abs(pt1-pt2), np.abs(pt3-pt2))
        alpha3 = angle(np.abs(pt2-pt3), np.abs(pt4-pt3))
        alpha4 = angle(np.abs(pt3-pt4), np.abs(pt0-pt4))

        return pt0, pt1, pt2, pt3, pt4

    def __split_cnt__(self, cnt):
        pt0 = cnt[0][0]
        pt1 = cnt[1][0]
        pt2 = cnt[2][0]
        pt3 = cnt[3][0]
        pt4 = cnt[4][0]
        return  pt0, pt1, pt2, pt3, pt4

def show_contours(cnt, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, cnt, -1, (0, 0, 255), 1)
    cv2.imshow(window_name, preview)

def draw_ball(ball, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.circle(preview, (int(ball[0][0]), int(ball[0][1])), int(ball[1]), (0, 255, 0), 1)
    cv2.imshow(window_name, preview)

def draw_balls(balls, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for ball in balls:
        cv2.circle(preview, (int(ball[0][0]), int(ball[0][1])), int(ball[1]), (0, 255, 0), 1)
    cv2.imshow(window_name, preview)

def draw_home_lines(lines, frame, window_name):
    preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.line(preview, (lines[0][0][0], lines[0][0][1]), (lines[0][1][0], lines[0][1][1]), (255, 0, 0), 1)
    cv2.line(preview, (lines[1][0][0], lines[1][0][1]), (lines[1][1][0], lines[1][1][1]), (255, 0, 0), 1)
    cv2.line(preview, (lines[2][0][0], lines[2][0][1]), (lines[2][1][0], lines[2][1][1]), (0, 255, 0), 1)
    cv2.line(preview, (lines[3][0][0], lines[3][0][1]), (lines[3][1][0], lines[3][1][1]), (0, 255, 0), 1)
    cv2.line(preview, (lines[4][0][0], lines[4][0][1]), (lines[4][1][0], lines[4][1][1]), (0, 0, 255), 1)
    cv2.imshow(window_name, preview)

def refining_corners(gray, corners, winSize):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cv2.cornerSubPix(gray, corners, winSize, (-1, -1), criteria)

def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = np.inner(pt1, pt2)
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)

    # return the angle in degrees
    return np.degrees(math.acos(inner_product/(len1*len2)))
