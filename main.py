import os
import cv2
import detect_home
import capture_ball
from cvinput import cvwindows
from parse_args import args
from kmeans import kmeans
from utils import Reader, Obj
from filtering import filter_img

params = Obj(
    useKmeans=False,
    kmeans_k=6
)

def main():
    camera = cvwindows.create('camera')

    reader = Reader()
    setUp_Reader(reader)

    home_tracking = []
    ball_tracking = []

    mog2 = cv2.BackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    #setting up detect_home and capture_ball params
    detect_home.setUp({"debugging":args.debugging})
    capture_ball.setUp({"debugging":args.debugging})

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

        #finding a list of homes
        homes = detect_home.get_homes(gray)
        if homes is None or len(homes) == 0:
            print reader.get_actualFrame()
            continue

        # keep the best home
        home_tracking.append(homes)

        # transform the frame

        # finding the ball
        centers, radiuses = capture_ball.get_ball(gray, mog2, kernel)
        if len(radiuses) > 0:
            ball_tracking.append((centers, radiuses))

        # draw home and the balls trajectory
        contours_img = frame.copy()
        cv2.drawContours(contours_img, homes.astype('int32'), -1, (0, 0, 255), 2)
        for j in range(len(centers)):
            cv2.circle(contours_img, (int(centers[j][0]), int(centers[j][1])), int(radiuses[j]), (0, 255, 0), 1)
        camera.show(contours_img)

        # if len(homes) > 1:
        #     cv2.waitKey(0)

    cvwindows.clear()

    # draw final result
    reader.restart_reading()
    frame = reader.read()

    for centers, radiuses in ball_tracking:
        for j in range(len(radiuses)):
            cv2.circle(frame, (int(centers[j][0]), int(centers[j][1])), int(radiuses[j]), (0, 255, 0), 1)
    cv2.imshow("tracking", frame)
    cv2.waitKey()

def setUp_Reader(reader):
    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/'
    params = {}
    params["folder_path"] = path
    reader.setUp(params)

def setUp(nparams):
    params.setattr(nparams)

if __name__ == "__main__":
    main()
