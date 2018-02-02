import cv2
from utils import Obj

params = Obj(
    blur_algorithm=cv2.medianBlur,
    win_size=5,
    iter_number=1
)

def filter_img(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = gray
    for _ in range(params.iter_number):
        blur = params.blur_algorithm(blur, params.win_size)
    return blur

def setUp(nparams):
    params.setattr(nparams)
