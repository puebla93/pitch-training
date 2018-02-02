import os
import time
from datetime import datetime
import cv2
import beep
from cvinput import cvwindows
from parse_args import args

def main():
    camera = cvwindows.create('camera')
    capture = cv2.VideoCapture(args.camera)
    set_props(capture, ["FRAME_WIDTH", "FRAME_HEIGHT", "FPS"], [320, 240, 187])

    show_fps = args.debugging
    save = False
    frames = []

    frames_count = 0
    t = datetime.now()
    start_time = datetime.now()
    adjust_image = False

    while cvwindows.event_loop():
        _, img = capture.read()
        camera.show(img)

        if (not adjust_image) and (datetime.now() - start_time).seconds >= 5:
            adjust_image = True
            gain = capture.get(cv2.cv.CV_CAP_PROP_GAIN)
            capture.set(cv2.cv.CV_CAP_PROP_GAIN, 0.0)
            # print capture.get(cv2.cv.CV_CAP_PROP_GAIN)
            # print
            # set_props(capture, ["GAIN"], [gain])
            # capture.set(18, capture.get(18))

        if show_fps and (datetime.now() - t).seconds >= 1:
            print frames_count
            frames_count = 0
            t = datetime.now()
        frames_count += 1

        if (not adjust_image) and cvwindows.last_key == ' ':
            print "wait for adjust image"
        elif cvwindows.last_key == ' ':
            beep.beep()
            save = not save
            if not save:
                print "done"
                write_frames(frames)
                beep.beep()
                frames = []
            else:
                print "recording frames..."
        if save:
            frames.append(img)
        # print capture.get(cv2.cv.CV_CAP_PROP_GAIN)
        # if (datetime.now() - start_time).seconds >= 6:
        #     break

def write_frames(frames):
    print "writing to disk..."
    folder_name = time.asctime()
    os.mkdir(folder_name)
    i = 0
    for frame in frames:
        file_name = folder_name + "/" + str(i) + ".png"
        cv2.imwrite(file_name, frame)
        i += 1
    print "done"

def set_props(capture, props, values):
    min_len = min(len(props), len(values))
    for i in range(min_len):
        capture.set(capPropId(props[i]), values[i])

def capPropId(prop):
    return getattr(cv2.cv, "CV_" + "CAP_PROP_" + prop)

if __name__ == "__main__":
    main()
