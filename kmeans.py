import os
import numpy as np
import cv2
from params import params
from parse_args import args

def kmeans(frame, K):
    Z = frame.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    tmp = center[label.flatten()]
    result_frame = tmp.reshape((frame.shape))

    return result_frame

if __name__ == '__main__':
    folder_path = os.listdir("videos")
    folder_path.sort()
    path = 'videos/' + folder_path[args.test_folder] + '/' + str(args.test_frame) + '.png'
    frame = cv2.imread(path, 0)

    kmeans_frame = kmeans(frame, params.kmeans_k)

    cv2.imshow('frame', frame)
    cv2.imshow('Kmeans', kmeans_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
