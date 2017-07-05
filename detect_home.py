import cv2
from cvinput import cvwindows

def main():
    camera = cvwindows.create('camera')
    frame = cv2.imread('0.png', 0)
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    _,thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)
    camera.show(frame)
    cv2.imshow('Thresh', thresh)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    while cvwindows.event_loop():
        pass

if __name__ == "__main__":
    main()
