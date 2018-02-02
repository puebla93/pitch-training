import os
import cv2
import detect_home
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

        # setUp detect_home params and finding a list of homes
        detect_home.setUp({"debugging":args.debugging})
        homes = detect_home.get_homes(gray)
        if homes is None or len(homes) == 0:
            print reader._actual_frame
            continue

        # keep the best home

        # transform the frame

        # find the ball

        contours_img = frame.copy()
        cv2.drawContours(contours_img, homes.astype('int32'), -1, (0, 0, 255), 2)
        camera.show(contours_img)

        if len(homes) > 1:
            cv2.waitKey(0)

def setUp_Reader(reader):
    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/'
    params = {}
    params["folder_path"] = path
    reader.setUp(params)

if __name__ == "__main__":
    main()
