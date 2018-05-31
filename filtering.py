import cv2
from utils import Obj

params = Obj(
    blur_algorithm=cv2.medianBlur,
    win_size=5,
    iter_number=1
)

def filter_img(frame):
    blur = frame.copy()
    for _ in range(params.iter_number):
        blur = params.blur_algorithm(blur, params.win_size)
    return blur

def setUp(nparams):
    params.setattr(nparams)
