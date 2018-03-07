import numpy as np
import cv2
from utils import Obj

params = Obj(
    debugging=False,
    transform_resolution=(1024, 600),
    size_homePercenct=1/6.
)

def get_home_square(home):
    # rect = cv2.minAreaRect(home.contour)
    # box = np.array(cv2.cv.BoxPoints(rect), dtype="float32")
    # return home.ordered_pts[1:]
    pts = []
    pt1 = home.ordered_pts[2] + (home.ordered_pts[2] - home.ordered_pts[1])*2
    pt4 = home.ordered_pts[3] + (home.ordered_pts[3] - home.ordered_pts[4])*2
    pts.append(pt1)
    pts.append(home.ordered_pts[3])
    pts.append(home.ordered_pts[4])
    pts.append(pt4)
    return home.ordered_pts[1:]
    # return pts

def homePlate_transform(frame, home):
	# obtain a square in a consistent points order
    square = get_home_square(home)

    # compute the destination points of the square in the perspective transform
    home_width = params.transform_resolution[1] * params.size_homePercenct
    x14 = params.transform_resolution[0]
    x23 = params.transform_resolution[0] - home_width
    y12 = (params.transform_resolution[1] + home_width)/2.
    y34 = (params.transform_resolution[1] - home_width)/2.
    # dst=np.array([
    #     [x14, y12],
    #     [x23, y12],
    #     [x23 , y34],
    #     [x14, y34]], dtype="float32")
    dst=np.array([
        [x14-home_width/2., y12],
        [x23, y12],
        [x23 , y34],
        [x14-home_width/2., y34]], dtype="float32")

	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(square, dst)
    warped = cv2.warpPerspective(frame, M, params.transform_resolution)
    
    if params.debugging:
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
        cv2.destroyWindow('Warped')
        print "TRANSFORM FRAME DONE!!!\n"        
    
	# return the warped frame
    return warped

def setUp(nparams):
    params.setattr(nparams)
