import os
import numpy as np
import cv2
import detect_homes
import transform
import capture_balls
from cvinput import cvwindows
from parse_args import args
from util.utils import Reader, Obj, show_contours, HomePlate, kmeans
from filtering import filter_img

class PitchTrainig():
    def __init__(self):
        self.camera = cvwindows.create('camera')

        self.reader = Reader()

        # setting up reader, detect_homes and capture_balls params
        self.__setUp_Reader__()
        detect_homes.setUp({"debugging":args.debugging})
        transform.setUp({"debugging":args.debugging})
        capture_balls.setUp({"debugging":args.debugging})

        self.useKmeans = False
        self.kmeans_k = 3
    
    def start(self):
        while cvwindows.event_loop():
            home = self.calibrateHome()

            PTM, user_homePlate_cnt = self.computeTransform(home)

            self.waitBalls(PTM)

            # self.__draw_result__()

    def calibrateHome(self):
        home_tracking = []
        while len(home_tracking) < 200:
            # reading a frame
            frame = self.reader.read()
            if frame is None:
                break

            # removing noise from image
            gray = filter_img(frame)

            # using kmeans on the image
            if self.useKmeans:
                gray = kmeans(frame, self.kmeans_k)
                if args.debugging:
                    cv2.imshow('kmeans', gray)
                    cv2.waitKey(0)

            # finding a list of homes
            contours = detect_homes.get_homes(gray)
            if contours is None or len(contours) == 0:
                print(self.reader.get_frameNumber())
                continue

            # keeping the best home
            home = self.__homeAVG__(contours)
            home_tracking.append(home)

            self.__drawHomes__(frame, contours)

            if len(contours) > 1:
                print("len = ", len(contours))
                print(self.reader.get_frameNumber())

        return HomePlate(self.__homeAVG__(home_tracking))

    def computeTransform(self, home):
        gray = filter_img(self.reader.actualFrame)
        PTM, user_homePlate_cnt = transform.homePlate_transform(gray, home)
        return PTM, user_homePlate_cnt

    def waitBalls(self, PTM):
        ball_tracking = []
        while True:
            # reading a frame
            frame = self.reader.read()
            if frame is None:
                break

            # removing noise from image
            gray = filter_img(frame)

            # using kmeans on the image
            if self.useKmeans:
                gray = kmeans(frame, self.kmeans_k)
                if args.debugging:
                    cv2.imshow('kmeans', gray)
                    cv2.waitKey(0)

            # transform the frame
            warped = cv2.warpPerspective(gray, PTM, transform.params.transform_resolution)

            # finding the ball
            balls = capture_balls.get_balls(warped)
            if len(balls) > 0:
                ball_tracking.append(balls)

    def __homeAVG__(self, homes):
        home = np.mean(homes, 0)
        return home

    def __drawHomes__(self, frame, contours):
        contours_img = frame.copy()
        cv2.drawContours(contours_img, contours.astype('int32'), -1, (0, 0, 255), 2)
        self.camera.show(contours_img)

    def __draw_result__(self, homePlate_cnt, ball_tracking, ball_func):
        user_img = cv2.cvtColor(np.zeros(params.transform_resolution, 'float32'), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(user_img, [homePlate_cnt.astype('int32')], -1, (255, 255, 255), -1)

        for balls in ball_tracking:
            for center, radius in balls:
                cv2.circle(user_img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), -1)
        
        return user_img

    def setUp(self, nparams):
        params.setattr(nparams)

    def __setUp_Reader__(self):
        folder_path = os.listdir("videos")
        folder_path.sort()
        path = 'videos/' + folder_path[args.test_folder] + '/'
        reader_params = {}
        reader_params["folder_path"] = path
        self.reader.setUp(reader_params)
