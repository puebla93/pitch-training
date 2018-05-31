import numpy as np
import cv2
from utils import Obj, HomePlate
from utils import show_contours

params = Obj(
    debugging=False,
    transform_resolution=(1024, 600),
    size_homePercenct=1./6,
    use_rectApprox = False
)

def homePlate_transform(frame, home):
	# obtain a square in a consistent points order
    square = get_home_square(home)

    # compute the destination points of the perspective transform square
    home_width = params.transform_resolution[1] * params.size_homePercenct
    x14 = params.transform_resolution[0] - params.transform_resolution[0] * .01
    x23 = x14 - home_width
    y12 = (params.transform_resolution[1] + home_width)/2.
    y34 = y12 - home_width
    dst=np.array([
        [x14, y12],
        [x23, y12],
        [x23 , y34],
        [x14, y34]], dtype="float32")

	# compute the perspective transform matrix
    PTM = cv2.getPerspectiveTransform(square, dst)

    # find the new home plate contour
    homePlate_cnt = np.array([
                    [x14, params.transform_resolution[1]/2.],
                    [x23 + home_width/2., y12],
                    [x23, y12],
                    [x23, y34],
                    [x23 + home_width/2., y34]])

    if params.debugging:
        warped = cv2.warpPerspective(frame, PTM, params.transform_resolution)
        cnt = square.reshape((4,1,2))
        show_contours([cnt.astype('int32')], frame, 'Square')
        cv2.imshow('Warped', warped)
        cv2.waitKey(0)
        cv2.destroyWindow('Warped')
        cv2.destroyWindow('Square')
        print "TRANSFORMING FRAME DONE!!!\n"
    
	# return the perspective transform matrix
    return PTM, homePlate_cnt

def get_home_square(home):
    pt1 = home.ordered_pts[2] + (home.ordered_pts[1] - home.ordered_pts[2]) * 2
    pt2 = home.ordered_pts[2]
    pt3 = home.ordered_pts[3]
    pt4 = home.ordered_pts[3] + (home.ordered_pts[4] - home.ordered_pts[3]) * 2
    square = np.array([pt1, pt2, pt3, pt4])

    if params.use_rectApprox:
        rect = cv2.minAreaRect(square.reshape((4, 1, 2)))
        box = np.array(cv2.cv.BoxPoints(rect), dtype="float32")
        return box
    return square

def setUp(nparams):
    params.setattr(nparams)
