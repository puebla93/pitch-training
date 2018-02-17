import math
import cv2
import numpy as np

class Obj(object):
    def __init__(self, **kwarg):
        self.setattr(kwarg)
    
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
        self.contour = cnt
        pts = np.array([pt[0] for pt in cnt])
        self.ordered_pts = self.__find_order__(pts)

    def __find_order__(self, pts):
        angles = []
        angles.append(angle(pts[4]-pts[0], pts[1]-pts[0]))
        angles.append(angle(pts[0]-pts[1], pts[2]-pts[1]))
        angles.append(angle(pts[1]-pts[2], pts[3]-pts[2]))
        angles.append(angle(pts[2]-pts[3], pts[4]-pts[3]))
        angles.append(angle(pts[3]-pts[4], pts[0]-pts[4]))

        if self.__rigth_angles_order__(angles):
            return pts
        elif self.__rigth_angles_order__(np.roll(angles, 1)):
            return np.roll(pts, 1)
        elif self.__rigth_angles_order__(np.roll(angles, 2)):
            return np.roll(pts, 2)
        elif self.__rigth_angles_order__(np.roll(angles, 3)):
            return np.roll(pts, 3)
        elif self.__rigth_angles_order__(np.roll(angles, 4)):
            return np.roll(pts, 4)
        return pts

    def __rigth_angles_order__(self, angles):
        # angles most to be approx [90, 135, 90, 90, 135]
        return np.allclose(angles, [90, 135, 90, 90, 135], 0, 5)

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

def get_dist(lines):
    dist = []
    for line in lines:
        dist.append(((line[0][0] - line[1][0])**2+(line[0][1] - line[1][1])**2)**.5)
    return dist

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
