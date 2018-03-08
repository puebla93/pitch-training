import os
import numpy as np
import cv2
import detect_homes
import transform
import capture_balls
from cvinput import cvwindows
from parse_args import args
from utils import Reader, Obj, show_contours, HomePlate, kmeans
from filtering import filter_img

params = Obj(
    useKmeans=False,
    kmeans_k=6,
    transform_resolution=(600, 1024)
)

def main2():
    from pitch_trainig import PitchTrainig
    pitchTrainig = PitchTrainig()
    while True:
        home = pitchTrainig.calibrateHome()

        PTM, user_homePlate_cnt = pitchTrainig.computeTransform(home)

        pitchTrainig.waitBalls(PTM)

def main():
    camera = cvwindows.create('camera')

    reader = Reader()
    setUp_Reader(reader)

    ball_tracking = []

    # setting up detect_homes and capture_balls params
    detect_homes.setUp({"debugging":args.debugging})
    transform.setUp({"debugging":args.debugging})
    capture_balls.setUp({"debugging":args.debugging})
    
    home = calibratingHome(reader)

    frame = reader.read()
    gray = filter_img(frame)
    PTM, new_homePlate_cnt = transform.homePlate_transform(gray, home)

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

        # transform the frame
        warped = cv2.warpPerspective(gray, PTM, transform.params.transform_resolution)

        # finding the ball
        balls = capture_balls.get_balls(warped)
        # balls = capture_balls.get_balls(gray)
        if len(balls) > 0:
            ball_tracking.append(balls)

        camera.show(frame)

    cvwindows.clear()

    # draw final result
    user_img = draw_result(new_homePlate_cnt, ball_tracking, None)
    cv2.imshow('RESULT', user_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def calibratingHome(reader):
    home_tracking = []
    while len(home_tracking) < 200:
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
        contours = detect_homes.get_homes(gray)
        if contours is None or len(contours) == 0:
            print reader.get_actualFrame()
            cv2.waitKey(0)
            continue

        # keep the best home
        home = homeAVG(contours)
        home_tracking.append(home)

        contours_img = frame.copy()
        cv2.drawContours(contours_img, contours.astype('int32'), -1, (0, 0, 255), 2)
        cv2.imshow('Homes', contours_img)
        cv2.waitKey(1)

        if len(contours) > 2:
            print "len = ", len(contours)
            print reader.get_actualFrame()
            cv2.waitKey(0)

    cv2.destroyWindow('Homes')
    return HomePlate(homeAVG(home_tracking))

def draw_result(homePlate_cnt, ball_tracking, ball_func):
    user_img = cv2.cvtColor(np.zeros(params.transform_resolution, 'float32'), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(user_img, [homePlate_cnt.astype('int32')], -1, (255, 255, 255), -1)

    for balls in ball_tracking:
        for center, radius in balls:
            cv2.circle(user_img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), -1)
    
    return user_img

def homeAVG(homes):
    home = np.mean(homes, 0)
    return home

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
