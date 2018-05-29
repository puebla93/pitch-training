import os
import cv2
import numpy as np
import json
from parse_args import args
import math

drawing = False # true if mouse is pressed
strike = False # if True, draw rectangle. Press 'm' to toggle to curve
fx, fy = -1,-1
sx, sy = -1,-1
frame = None
showed_frame = None

def save(data, file_path):
    with open(file_path, 'w') as outfile:
        try:
            json.dump(data, outfile, indent=4)
        except UnicodeDecodeError:
            pass

def load(file_path):
    try:
        with open(file_path) as json_file:
            return json.load(json_file)
    except IOError:
        return {}

# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global fx, fy, sx, sy, drawing, showed_frame

    if event == cv2.EVENT_LBUTTONDOWN and not drawing:
        showed_frame = frame.copy()
        drawing = True
        fx, fy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        sx, sy = x, y
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def move_rectangle(k):
    global fx, fy, sx, sy, showed_frame

    if k == 81:
        fx -= 1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 82:
        fy -= 1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 83:
        sx +=1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 84:
        sy += 1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 180:
        fx += 1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 184:
        fy += 1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 182:
        sx -=1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k == 178:
        sy -= 1
        showed_frame = frame.copy()
        cv2.rectangle(showed_frame, (fx, fy), (sx, sy), (0,255,0), 1)
    elif k != 255:
        print k

def saveBallPosition(_frame, frameName):
    global frame, showed_frame, strike

    folder_path = os.listdir("pelota/full_HD(60fps)")
    folder_path.sort()
    folder_name = folder_path[args.test_folder]
    test = 'fullHD/'
    file_path = 'data_set/' + test + "test/" + folder_name + ".json"
    data = load(file_path)

    frame = _frame
    showed_frame = frame.copy()
    cv2.namedWindow(frameName)
    cv2.setMouseCallback(frameName,draw_rectangle)

    while(1):
        cv2.imshow(frameName, showed_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            strike = not strike
        elif k == ord('s'):
            center = (sx + fx)/2., (sy+fy)/2.
            radius = abs(center[1] - fy)
            data[frameName] = [center, radius]
            data["strike"] = strike
            save(data, file_path)
            print "frame " + str(frameName) + " saved"
        elif k == ord('q'):
            break
        else:
            move_rectangle(k)

    cv2.destroyAllWindows()
