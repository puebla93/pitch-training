import cv2

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
