import os
import math
import cv2
import json
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
        elif len(self.params.frameNames) <= self._frameNumber:
            return None
        elif self.params.read_from == "folder":
            path = self.params.folder_path + self.params.frameNames[self._frameNumber]
            self.actualFrame = cv2.imread(path)
            # self.rectify_frame()
        return self.actualFrame

    def rectify_frame(self):
        mtx, dist = self.load_calibrationMatrix('calibration.json')
        h,  w = self.actualFrame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv2.undistort(self.actualFrame, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        self.actualFrame = dst[y:y+h, x:x+w]

    def load_calibrationMatrix(self, path):
        with open(path) as f:
            loadeddict = json.load(f)

        mtxloaded = np.array(loadeddict.get('camera_matrix'))
        distloaded = np.array(loadeddict.get('dist_coeff'))
        
        return mtxloaded, distloaded

    def restart_reading(self):
        self.actualFrame = None
        if self.params.read_from == "camera":
            self.capture.release()
            self.capture = cv2.VideoCapture(self.params.camera_index)
        elif self.params.read_from == "folder":
            self._frameNumber = 0

    def get_frameName(self):
        if len(self.params.frameNames) > self._frameNumber:
            return self.params.frameNames[self._frameNumber]
        else:
            return "-1"

    def setUp(self, nparams):
        self.params.setattr(nparams)
        if nparams.has_key("read_from") and nparams["read_from"] == "camera" and self.capture is None:
            self.capture = cv2.VideoCapture(self.params.camera_index)

    def __setDefaults__(self):
        listOfFrames = os.listdir("videos/Tue Jul  4 13:26:23 2017/")
        listOfFrames.sort(key=lambda frameName: int(frameName[:-4]))
        self.params = Obj(
            read_from="folder",
            folder_path="videos/Tue Jul  4 13:26:23 2017/",
            frameNames=listOfFrames,
            camera_index=0
            )

class HomePlate():
    def __init__(self, cnt):
        # self.contour = cnt
        pts = [pt[0] for pt in cnt]
        self.ordered_pts = self.__find_order__(pts)
        self.contour = self.ordered_pts.reshape((5,1,2))

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
            return np.array(pts)
        elif indexes[0] + indexes[1] == 3:
            return np.array(pts[4:] + pts[:4])
        else:
            roll = (indexes[0] + indexes[1])/2
            rolled_pts = np.array(pts[roll:] + pts[:roll])
            return rolled_pts

class Ball:
    def __init__(self, center, radius, capture_frame):
        self.center = center
        self.radius = radius
        self.capture_frame = capture_frame

class QuadraticLeastSquaresModel:
    def __init__(self):
        self.func = lambda x, a, b, c : a+b*x+c*x*x

    def fit(self, data):
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        x0 = np.array([0.0, 0.0, 0.0])
        sigma = np.ones(data.shape[0], 'float32')
        Yvalues, _ = optimization.curve_fit(self.func, x, y, x0, sigma)
        Zvalues, _ = optimization.curve_fit(self.func, x, z, x0, sigma)
        return Yvalues, Zvalues

        # A = np.array([(19,20,24), (10,40,28), (10,50,31)])

        # def func(data, a, b):
        #     return data[:,0]*data[:,1]*a + b

        # guess = (1,1)
        # params, pcov = optimize.curve_fit(func, A[:,:2], A[:,2], guess)
        # print(params)
        # # [ 0.04919355  6.67741935]

    def get_error(self, data, model):
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        Ymodel, Zmodel = model
        y_fit = self.func(x, Ymodel[0], Ymodel[1], Ymodel[2])
        z_fit = self.func(x, Zmodel[0], Zmodel[1], Zmodel[2])
        y_error = (y - y_fit)**2
        z_error = (z - z_fit)**2
        err_per_point = y_error + z_error# sum squared error per row
        return err_per_point

def show_contours(cnt, frame, window_name):
    if len(frame.shape) == 2:
        preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3:
        preview = frame.copy()
    cv2.drawContours(preview, cnt, -1, (0, 0, 255), 1)
    cv2.imshow(window_name, preview)

def draw_ball(ball, frame, window_name):
    if len(frame.shape) == 2:
        preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3:
        preview = frame.copy()
    cv2.circle(preview, (int(ball.center[0]), int(ball.center[1])), int(ball.radius), (0, 255, 0), 1)
    cv2.imshow(window_name, preview)

def draw_balls(balls, frame, window_name):
    if len(frame.shape) == 2:
        preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3:
        preview = frame.copy()
    for ball in balls:
        cv2.circle(preview, (int(ball.center[0]), int(ball.center[1])), int(ball.radius), (0, 255, 0), 1)
    cv2.imshow(window_name, preview)

def draw_home_lines(lines, frame, window_name):
    if len(frame.shape) == 2:
        preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3:
        preview = frame.copy()
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
    contours = map(lambda home: home.contour, homes)
    contour = np.mean(contours, 0)
    return HomePlate(contour)

def draw_finalResult(homePlate_cnt, balls, img_resolution, wasStrike, velocity):
    user_img = cv2.cvtColor(np.zeros(img_resolution, 'float32'), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(user_img, [homePlate_cnt.astype('int32')], -1, (255, 255, 255), -1)

    ballColor, pitch = ((0, 255, 0), 'STRIKE '+velocity) if wasStrike else ((0, 0, 255), 'BALL '+velocity)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(user_img, pitch, (10,50), font, 2, ballColor, 2)

    for ball in balls:
        cv2.circle(user_img, (int(ball.center[0]), int(ball.center[1])), int(ball.radius), ballColor, -1)

    return user_img

def draw_strikeZone(img_resolution, ballFunc, wasStrike, velocity):
    YballFunc, ZballFunc = ballFunc
    Yfunc = lambda x: YballFunc[0] + YballFunc[1]*x + YballFunc[2]*x**2
    Zfunc = lambda x: ZballFunc[0] + ZballFunc[1]*x + ZballFunc[2]*x**2

    user_img = cv2.cvtColor(np.zeros(img_resolution, 'float32'), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(user_img, (250, int(117*2.31)), (350, int(186*2.31)), (255, 255, 255), 1)
    cv2.line(user_img, (350, img_resolution[0]-1), (250, img_resolution[0]-1), (255, 255, 255), 3)

    ballColor, pitch = ((0, 255, 0), 'STRIKE '+velocity) if wasStrike else ((0, 0, 255), 'BALL '+velocity)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(user_img, pitch, (10,50), font, 2, ballColor, 2)

    x = int(Yfunc(913))
    ball_diameter_pixels = 2.86/17*100
    ball_pixels = Zfunc(913)*2
    ball_high = 225 - (225*ball_diameter_pixels/ball_pixels)
    y = int((225-ball_high)*2.31)
    cv2.circle(user_img, (x,y), int(7*2.31), ballColor, -1)
    
    return user_img

def draw_SideTrajectory(img_resolution, balls, ZballFunc, wasStrike, velocity):
    Zfunc = lambda x: ZballFunc[0] + ZballFunc[1]*x + ZballFunc[2]*x**2
    user_img = cv2.cvtColor(np.zeros(img_resolution, 'float32'), cv2.COLOR_GRAY2BGR)

    ballColor, pitch = ((0, 255, 0), 'STRIKE '+velocity) if wasStrike else ((0, 0, 255), 'BALL '+velocity)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(user_img, pitch, (10,50), font, 2, ballColor, 2)

    ball_diameter_pixels = 2.86/17*100
    for ball in balls:
        ball_pixels = Zfunc(ball.center[0])*2
        ball_high = 225 - (225*ball_diameter_pixels/ball_pixels)
        y = int((225-ball_high)*2.31)
        cv2.circle(user_img, (int(ball.center[0]), y), int(7*2.31), ballColor, -1)
    
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

def plot_fit(all_balls, n_balls):
    points = map(lambda ball: np.array([ball.center[0], ball.center[1], ball.radius]), all_balls)
    points = np.array(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    points = map(lambda ball: np.array([ball.center[0], ball.center[1], ball.radius]), n_balls)
    points = np.array(points)
    n_x, n_y, n_z = points[:, 0], points[:, 1], points[:, 2]

    plt.plot(x, y, '-o')
    plt.plot(x, z, '-o')
    plt.plot(n_x, n_y, '-o')
    plt.plot(n_x, n_z, '-o')

    plt.xlim(0, 1024)
    plt.ylim(0, 600)
    plt.show()

def fit_velocity(data):
    model = lambda x, a, b: a + b*x
    # model = lambda x, a, b, c: a + b*x + c*x*x

    x, y = data[:, 0], data[:, 1]
    x0 = np.array([0.0, 0.0])
    # x0 = np.array([0.0, 0.0, 0.0])
    sigma = np.ones(data.shape[0], 'float32')
    values, _ = optimization.curve_fit(model, x, y, x0, sigma)
    func = lambda x: values[0] + values[1]*x
    # func = lambda x: values[0] + values[1]*x + values[2]*x*x
    return func

def plot_velocity(points, func):
    x, y = points[:, 0], points[:, 1]

    n_y = func(x)

    plt.scatter(x, y, c='b', label='velocidad en cada frame')
    # plt.plot(x, z, '-o')
    plt.plot(x, n_y, c='r', lw=2, label='aprox de la velocidad')
    # plt.plot(n_x, n_z, '-o')

    # plt.xlim(0, 1024)
    # plt.ylim(0, 600)
    plt.xlabel('frames')
    plt.ylabel('velocidad en MPH')
    plt.legend()
    plt.show()
