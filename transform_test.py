import cv2
import numpy as np
import detect_homes
import transform
from filtering import filter_img
from utils import Obj, HomePlate
from cvinput import cvwindows
from parse_args import args

def main():
    detect_homes.setUp({"debugging":args.debugging})
    transform.setUp({"debugging":args.debugging})

    camera = cvwindows.create('camera')
    transform_camera = cvwindows.create('transform')

    capture = cv2.VideoCapture(args.camera)
    # set_props(capture, ["FRAME_WIDTH", "FRAME_HEIGHT", "FPS"], [320, 240, 187])
    set_props(capture, ["FRAME_WIDTH", "FRAME_HEIGHT"], [320, 240])

    while cvwindows.event_loop():
        _, frame = capture.read()
        camera.show(frame)
        
        # removing noise from image    
        gray = filter_img(frame)

        # finding a list of homes
        contours = detect_homes.get_homes(gray)
        if contours is None or len(contours) == 0:
            user_img = cv2.cvtColor(np.zeros((600, 1024), 'float32'), cv2.COLOR_GRAY2BGR)
            transform_camera.show(user_img)
        else:
            mean = np.mean(contours, 0)
            home = HomePlate(mean)

            PTM, new_homePlate_cnt = transform.homePlate_transform(gray, home)
            warped = cv2.warpPerspective(gray, PTM, transform.params.transform_resolution)        

            transform_camera.show(warped)

            if args.debugging:
                cnt = np.reshape(home.ordered_pts, (5,1,2))

                pts = frame.copy()
                cv2.drawContours(pts, cnt.astype('int32'), 0, (0, 0, 255), 2)
                cv2.drawContours(pts, cnt.astype('int32'), 1, (0, 255, 0), 2)
                cv2.drawContours(pts, cnt.astype('int32'), 2, (255, 0, 0), 2)
                cv2.drawContours(pts, cnt.astype('int32'), 3, (255, 0, 255), 2)
                cv2.drawContours(pts, cnt.astype('int32'), 4, (0, 255, 255), 2)
                cv2.imshow('a', pts)

                img = frame.copy()
                cv2.drawContours(img, [home.contour.astype('int32')], -1, (0, 0, 255), 2)
                cv2.imshow('a2', img)

                user_img = cv2.cvtColor(np.zeros((600, 1024), 'float32'), cv2.COLOR_GRAY2BGR)
                cv2.drawContours(user_img, [new_homePlate_cnt.astype('int32')], -1, (255, 255, 255), -1)
                cv2.imshow('user_img', user_img)

                cv2.waitKey(0)
        
        if args.debugging:
            cv2.destroyWindow('a')
            cv2.destroyWindow('a2')
            cv2.destroyWindow('user_img')

    cvwindows.clear()

def set_props(capture, props, values):
    min_len = min(len(props), len(values))
    for i in range(min_len):
        capture.set(capPropId(props[i]), values[i])

def capPropId(prop):
    return getattr(cv2.cv, "CV_" + "CAP_PROP_" + prop)

if __name__ == "__main__":
    main()
