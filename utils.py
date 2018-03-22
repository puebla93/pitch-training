import math
import cv2
import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt

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
        self._frameNumber = -1
        self.actualFrame = None
        self.__setDefaults__()
        self.capture = None
        if self.params.read_from == "camera":
            self.capture = cv2.VideoCapture(self.params.camera_index)

    def read(self):
        self._frameNumber += 1
        if self.params.read_from == "camera":
            self.actualFrame = self.capture.read()[1]
        if self.params.read_from == "folder":
            path = self.params.folder_path + str(self._frameNumber) + self.params.frame_type
            self.actualFrame = cv2.imread(path)
        return self.actualFrame

    def restart_reading(self):
        self.actualFrame = None
        if self.params.read_from == "camera":
            self.capture.release()
            self.capture = cv2.VideoCapture(self.params.camera_index)
        elif self.params.read_from == "folder":
            self._frameNumber = 0

    def setUp(self, nparams):
        self.params.setattr(nparams)
        if nparams.has_key("read_from") and nparams["read_from"] == "camera" and self.capture is None:
            self.capture = cv2.VideoCapture(self.params.camera_index)

    def get_frameNumber(self):
        return self._frameNumber

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
        pts = [pt[0] for pt in self.contour]
        self.ordered_pts = self.__find_order__(pts)

    def __find_order__(self, pts):
        angles = []
        angles.append(angle(pts[4]-pts[0], pts[1]-pts[0]))
        angles.append(angle(pts[0]-pts[1], pts[2]-pts[1]))
        angles.append(angle(pts[1]-pts[2], pts[3]-pts[2]))
        angles.append(angle(pts[2]-pts[3], pts[4]-pts[3]))
        angles.append(angle(pts[3]-pts[4], pts[0]-pts[4]))

        indexes = [0, 1, 2, 3, 4]
        indexes.sort(key=(lambda i: angles[i]), reverse=True)

        if indexes[0] + indexes[1] == 5:
            return pts
        elif indexes[0] + indexes[1] == 3:
            return pts[4:] + pts[:4]
        else:
            roll = (indexes[0] + indexes[1])/2
            rolled_pts = pts[roll:] + pts[:roll]
            return rolled_pts

class Ball:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

class QuadraticLeastSquaresModel:
    def __init__(self):
        self.func = lambda x, a, b, c : a+b*x+c*x*x

    def fit(self, data):
        all_centers = np.array(map(lambda b: b.center, data))
        x, y = all_centers[:, 0], all_centers[:, 1]
        x0 = np.array([0.0, 0.0, 0.0])
        sigma = np.ones(data.shape[0], 'float32')
        values, _ = optimization.curve_fit(self.func, x, y, x0, sigma)
        return values

    def get_error(self, data, model):
        all_centers = np.array(map(lambda b: b.center, data))
        x, y = all_centers[:, 0], all_centers[:, 1]
        y_fit = self.func(x, model[0], model[1], model[2])
        err_per_point = (y - y_fit)**2 # sum squared error per row
        return err_per_point

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

def angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    inner_product = np.inner(v1, v2)
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)

    # return the angle in degrees
    return np.degrees(math.acos(inner_product/(len1*len2)))

def homeAVG(homes):
    home = np.mean(homes, 0)
    return home

def draw_finalResult(homePlate_cnt, balls, img_resolution, ballFunc):
    user_img = cv2.cvtColor(np.zeros(img_resolution, 'float32'), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(user_img, [homePlate_cnt.astype('int32')], -1, (255, 255, 255), -1)

    meanRadius = int(np.mean(map(lambda b: b.radius, balls)))
    func = lambda x: ballFunc[0] + ballFunc[1]*x + ballFunc[2]*x**2
    start, stop, step = meanRadius, img_resolution[1], meanRadius*2
    for i in range(start, stop, step):
        cv2.circle(user_img, (i, int(func(i))), meanRadius, (0, 255, 0), -1)
    # for ball in balls:
        # cv2.circle(user_img, (int(ball.center[0]), int(ball.center[1])), int(ball.radius), (0, 255, 0), -1)
        # cv2.circle(user_img, (int(ball.center[0]), int(ball.center[1])), meanRadius, (0, 255, 0), -1)
    
    return user_img

def kmeans(frame, K):
    Z = frame.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    tmp = center[label.flatten()]
    result_frame = tmp.reshape((frame.shape))

    return result_frame

def plot_fit(balls_tracked, n_balls):    
    all_balls = np.array([ball for balls in balls_tracked for ball in balls])
    all_balls = np.array(map(lambda b: b.center, all_balls))
    x, y = all_balls[:, 0], all_balls[:, 1]

    all_balls = np.array(map(lambda b: b.center, n_balls))
    n_x, n_y = all_balls[:, 0], all_balls[:, 1]

    plt.plot(x, y, '-o')
    plt.plot(n_x, n_y, '-o')

    plt.xlim(0, 1024)
    plt.ylim(0, 600)
    plt.show()
