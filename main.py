import os
import numpy as np
import cv2
import detect_home
import transform
import capture_ball
from cvinput import cvwindows
from parse_args import args
from utils import Reader, Obj, show_contours, HomePlate, kmeans
from filtering import filter_img

params = Obj(
    useKmeans=False,
    kmeans_k=6,
    transform_resolution=(600, 1024)
)

def main():
    camera = cvwindows.create('camera')

    reader = Reader()
    setUp_Reader(reader)

    home_tracking = []
    ball_tracking = []

    # setting up detect_home and capture_ball params
    # detect_home.setUp({"debugging":args.debugging})
    transform.setUp({"debugging":args.debugging})
    # capture_ball.setUp({"debugging":args.debugging})

    # reader._actual_frame = 473
    # loop
    while cvwindows.event_loop():
        # reading a frame
        frame = reader.read()
        if frame is None:
            break

        # removing noise from image
        gray = filter_img(frame)

        # using kmeans on the image
        if params.useKmeans:
            gray = kmeans(frame, params.kmeans_k)
            if args.debugging:
                cv2.imshow('kmeans', gray)
                cv2.waitKey(0)

        # finding a list of homes
        contours = detect_home.get_homes(gray)
        if contours is None or len(contours) == 0:
            print reader.get_actualFrame()
            continue

        # keep the best home
        home = homeAVG(contours)
        home_tracking.append(home)

        # transform the frame
        warped = transform.homePlate_transform(gray, home)

        # finding the ball
        # balls = capture_ball.get_ball(warped)
        balls = capture_ball.get_ball(gray)
        if len(balls) > 0:
            ball_tracking.append(balls)

        # draw home and the balls trajectory
        contours_img = frame.copy()
        cv2.drawContours(contours_img, contours.astype('int32'), -1, (0, 0, 255), 2)
        for center, radius in balls:
            cv2.circle(contours_img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
        camera.show(contours_img)

        # if len(homes) > 1:
        #     print "len = ", len(homes)
        #     print reader.get_actualFrame()
        #     cv2.waitKey(0)

    cvwindows.clear()

    # draw final result
    # draw_result()

    reader.restart_reading()
    frame = reader.read()

    for balls in ball_tracking:
        for center, radius in balls:
            cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
    cv2.imshow("tracking", frame)
    cv2.waitKey()

    cv2.destroyAllWindows()

def draw_result(home_tracking, ball_tracking, ball_func):
    result = np.zeros(params.transform_resolution)
    home = homeAVG(home_tracking)
    cv2.drawContours(result, [home.astype('int32')], -1, (255, 255, 255), cv2.cv.CV_FILLED)
    for balls in ball_tracking:
        for center, radius in balls:
            cv2.circle(result, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
    cv2.imshow('RESULT', result)

def homeAVG(homes):
    home = np.mean(homes, 0)
    return HomePlate(home)

def setUp_Reader(reader):
    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/'
    reader_params = {}
    reader_params["folder_path"] = path
    reader.setUp(reader_params)

def setUp(nparams):
    params.setattr(nparams)

if __name__ == "__main__":
    main()
