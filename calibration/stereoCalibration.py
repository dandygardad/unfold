# Camera Calibration using Chessboard and OpenCV
# for ``unfold`` by Dandy Garda

# Import some libraries
import numpy as np
import cv2
import glob
import sys
import json

# Initialize chessboard and dimension
chessboard = (9, 6)
dim = (640, 480)

# Unix time value as the name of folder for chessboard on terminal
pathImg = sys.argv[1]

# Get square size from changeData.json
f = open('changeData.json')
squareSizeJson = json.load(f)
f.close()

squareSizeJson = squareSizeJson['stereoCalibration']['squareSize']

print("\n== GENERATE STEREO MAP FOR UNDISTORT AND RECTIFY STEREO CAMERA ==\n``unfold`` by dandy garda\n")

# CAMERA CALIBRATION STARTS HERE

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def matrixValues(cam_matrix, img_dim, apertureWidth=0, apertureHeight=0):
    fov_x, fov_y, focal_len, _, _ = cv2.calibrationMatrixValues(cam_matrix, img_dim, apertureWidth, apertureHeight)
    print("FOV x: " + str(fov_x))
    print("FOV y: " + str(fov_y))
    print("Focal Length: " + str(focal_len))

def manualFov(cam_matrix, dim):
    fov_x = np.rad2deg(2 * np.arctan2(dim[0], 2 * cam_matrix[0][0]))
    print("FOV x: " + str(fov_x))

# Function to camera calibration
def cameraCalibration(pathLeft, pathRight, squareSize, chessWidth=9, chessHeight=6):
    # FIND CHESSBOARD CORNERS (OBJECT POINTS AND IMAGE POINTS)
    # OBJECT POINTS = 3D POINT REAL WORLD
    # IMAGE POINTS = 2D POINT IN IMAGE
    
    # Prepare object points
    objp = np.zeros((chessWidth * chessHeight, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessWidth, 0:chessHeight].T.reshape(-1, 2)

    objp = objp * squareSize

    # Array to store object points and image points from all the images
    objpoints = [] # 3D point in real world space
    imgpointsL = [] # 2D point in image plane
    imgpointsR = [] # 2D point in image plane

    # Import images from generateCalibration
    imagesLeft = glob.glob(pathLeft)
    imagesRight = glob.glob(pathRight)

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, (chessWidth, chessHeight), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (chessWidth, chessHeight), None)

        # If found, add object points, image points (after refining them)
        if retL & retR == True:
            objpoints.append(objp)

            # Left
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            # Right
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, (chessWidth, chessHeight), cornersL, retL)
            cv2.imshow('Image Left', imgL)

            cv2.drawChessboardCorners(imgR, (chessWidth, chessHeight), cornersR, retR)
            cv2.imshow('Image Right', imgR)

            cv2.waitKey(1000)

    cv2.destroyAllWindows()



    # CALIBRATION (after we get obj points and image points)
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, dim, None, None)
    print("RMSE Left Camera: " + str(retL))

    # Get field of view and focal length
    # manualFov(cameraMatrixL, dim)
    matrixValues(cameraMatrixL, dim)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, dim, None, None)
    print("\nRMSE Right Camera: " + str(retR))
    matrixValues(cameraMatrixR, dim)
    
    # Get field of view and focal length
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



    # STEREO VISION CALIBRATION
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camera matrixes so that only Rot, Trns, Emat, Fmat are calculated
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between two cameras and calculate Essential
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    

    # STEREO RECTIFICATION
    rectifyScale = 1
    
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0,0))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixL, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)


    # Save stereo map
    cv_file = cv2.FileStorage('calibration\\stereoMap.xml', cv2.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x', stereoMapL[0])
    cv_file.write('stereoMapL_y', stereoMapL[1])
    cv_file.write('stereoMapR_x', stereoMapR[0])
    cv_file.write('stereoMapR_y', stereoMapR[1])

    cv_file.release()
    
    print("\nSaved the parameters as 'stereoMap.xml'!")


import os

# RUN IT
# change the path if there is new images for calibration
cameraCalibration(os.getcwd() + '\\calibration\\images\\'+ pathImg + '\\left\\*.jpg', os.getcwd() + '\\calibration\\images\\' + pathImg + '\\right\\*.jpg', squareSizeJson, chessboard[0], chessboard[1])
