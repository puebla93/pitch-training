import numpy as np
import cv2
from utils import Obj

params = Obj(
    debugging=False,
    transform_resolution=(1024, 600),
    dst=np.array([
        [950., 275.],
        [1000., 275.],
        [1000., 325.],
        [950., 325.]], dtype="float32")
)

def get_home_square(pts):
    pass
    return pts

def homePlate_transform(image, pts):
	# obtain a square in a consistent points order
    square = get_home_square(pts)

	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(square, params.dst)
    warped = cv2.warpPerspective(image, M, params.transform_resolution)
    
    if params.debugging:
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
    
	# return the warped image
    return warped

def setUp(nparams):
    params.setattr(nparams)
