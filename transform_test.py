import cv2
import numpy as np
import detect_homes
import transform
from filtering import filter_img
from util.utils import Obj, HomePlate, homeAVG
from cvinput import cvwindows
from parse_args import args
import beep

def main():
    resolution = (640, 480)
    detect_homes.setUp({"debugging":args.debugging})
    transform.setUp({"debugging":args.debugging, "transform_resolution":resolution, "size_homePercenct":1./4})

    camera = cvwindows.create('camera')
    transform_camera = cvwindows.create('transform')

    capture = cv2.VideoCapture(args.camera)
    set_props(capture, ["FRAME_WIDTH", "FRAME_HEIGHT", "FPS"], [resolution[0], resolution[1], 5])
    # set_props(capture, ["FRAME_WIDTH", "FRAME_HEIGHT", "FPS"], [320, 240, 5])

    save = False
    frames = []
    dual_frames = []

    while cvwindows.event_loop():
        _, frame = capture.read()
        
        # removing noise from image
        blur = filter_img(frame)

        if cvwindows.last_key == ' ':
            beep.beep()
            save = not save

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # finding a list of homes
        homes = detect_homes.get_homes(gray)
        if homes is None or len(homes) == 0:
            # user_img = cv2.cvtColor(np.zeros((600, 1024), 'float32'), cv2.COLOR_GRAY2BGR)
            # user_img = np.zeros((480, 640*2))
            user_img = np.zeros((resolution[1], resolution[0]), 'float32')
            transform_camera.show(user_img)
            if save:
                frames.append(frame)
        else:
            # keep the best home
            home = homeAVG(homes)

            PTM, new_homePlate_cnt = transform.homePlate_transform(gray, home)
            warped = cv2.warpPerspective(frame, PTM, transform.params.transform_resolution)        

            res = cv2.resize(warped, resolution, interpolation = cv2.INTER_CUBIC)
            # transform_camera.show(warped)
            transform_camera.show(res)

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

            cv2.drawContours(frame, [home.contour.astype('int32')], -1, (0, 0, 255), 2)
            dual_frame = np.zeros((resolution[1], resolution[0]*2, 3), 'float32')
            dual_frame[:, :resolution[0], :] = frame
            dual_frame[:, resolution[0]:, :] = res
            dual_frames.append(dual_frame)
            # transform_camera.show(dual_frame)

        camera.show(frame)
        if args.debugging:
            cv2.destroyWindow('a')
            cv2.destroyWindow('a2')
            cv2.destroyWindow('user_img')

    cvwindows.clear()

    if len(frames) != 0:
        write_frames(frames)
    if len(dual_frames) != 0:
        i = 0
        for dual_frame in dual_frames:
            file_name = 'presentation/' + str(i) + ".jpg"
            cv2.imwrite(file_name, dual_frame)
            i += 1
        print("done")

def write_frames(frames):
    import os
    print("writing to disk...")
    imgs_name = os.listdir("./videos/transform_test/")
    imgs_name.sort()
    i = len(imgs_name)
    for frame in frames:
        file_name = "./videos/transform_test/" + str(i) + ".png"
        cv2.imwrite(file_name, frame)
        i += 1
    print("done")

def set_props(capture, props, values):
    min_len = min(len(props), len(values))
    for i in range(min_len):
        capture.set(capPropId(props[i]), values[i])

def capPropId(prop):
    return getattr(cv2.cv, "CV_" + "CAP_PROP_" + prop)

if __name__ == "__main__":
    main()
