import numpy as np
import cv2
import json
import glob
from util.utils import Obj

params = Obj(
    chessboard_size=(9,6),
    winSize=(11,11)
)

def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((params.chessboard_size[1]*params.chessboard_size[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:params.chessboard_size[0],0:params.chessboard_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('chessboard_images/*.jpg')
    found = 1
    images.sort(key=lambda image: int(image[-5:-4]) if image[-6] == '/' else int(image[-6:-4]))
    for fname in images:
        img = cv2.imread(fname) # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, params.chessboard_size,None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
            cv2.cornerSubPix(gray,corners,params.winSize,(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, params.chessboard_size, corners, ret)
            cv2.imwrite('chessboard_images/'+str(found)+'.png', img)
        else:
            to_print = fname[-5:] if fname[-6] == '/' else fname[-6:]
            print(to_print)

        found += 1
        cv2.imshow('img', img)
        cv2.waitKey(0)
        
    # When everything done
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(objpoints)))

    return mtx, dist

def save_matrix(mtx, dist, path):
    # It's very important to transform the matrix to list.

    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

    with open(path, "w") as f:
        json.dump(data, f)

def load_matrix(path):
    with open(path) as f:
        loadeddict = json.load(f)

    mtxloaded = loadeddict.get('camera_matrix')
    distloaded = loadeddict.get('dist_coeff')
    
    return mtxloaded, distloaded

def save_chessboard_images():
    camera_index = args.camera
    cap = cv2.VideoCapture(camera_index)
    i = 0
    while i < 30:
        ret, img = cap.read()
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            cv2.imwrite('chessboard_images/'+str(i)+'.jpg', img)
            i += 1
        cv2.imshow('img', img)

if __name__ == '__main__':
    # save_chessboard_images()
    mtx, dist = calibrate()
    save_matrix(mtx, dist, 'calibration.json')
