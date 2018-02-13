# import the necessary packages
import numpy as np
import cv2
from utils import Obj

params = Obj(
    debugging=False,
    transform_resolution=(1024, 600)
)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
    return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
    rect = order_points(pts)

	# construct the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points in the
	# top-left, top-right, bottom-right, and bottom-left order
    # dst = np.array([
    #    	[0, 0],
    #    	[maxWidth - 1, 0],
    #     [maxWidth - 1, maxHeight - 1],
    #     [0, maxHeight - 1]], dtype="float32")
    dst = np.float32(pts)

	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, params.transform_resolution)

    if params.debugging:
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)

	# return the warped image
    return warped

def setUp(nparams):
    params.setattr(nparams)
