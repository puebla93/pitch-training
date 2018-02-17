import numpy as np
import cv2
from utils import Obj

params = Obj(
    debugging=False,
    transform_resolution=(1024, 600),
    # dst=np.array([
    #     [950., 275.],
    #     [1000., 275.],
    #     [1000., 325.],
    #     [950., 325.]], dtype="float32")
    dst=np.array([
        [975, 325],
        [950., 325.],
        [950., 275.],
        [975, 275]], dtype="float32")
)

def get_home_square(home):
    rect = cv2.minAreaRect(home.contour)
    box = np.array(cv2.cv.BoxPoints(rect), dtype="float32")
    return home.ordered_pts[1:]
    # pts = []
    # pts.append(home.ordered_pts[3])
    # pts.append(home.ordered_pts[])
    # return pts

def homePlate_transform(frame, home):
	# obtain a square in a consistent points order
    square = get_home_square(home)

	# compute the perspective transform matrix and then apply it
    # M = cv2.getPerspectiveTransform(square, params.dst)
    M = cv2.getPerspectiveTransform(square, params.dst)
    warped = cv2.warpPerspective(frame, M, params.transform_resolution)
    
    if params.debugging:
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
        cv2.destroyWindow('Warped')
        print "\nTRANSFORM FRAME DONE!!!\n"        
    
	# return the warped frame
    return warped

def setUp(nparams):
    params.setattr(nparams)
