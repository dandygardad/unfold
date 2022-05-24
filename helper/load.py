# Functions to help stereo camera running and calibrated
# for ``unfold`` by dandy garda

import cv2
import os
from helper.general import originalDimCheck, errorMessage

# Load Stereo Camera
def stereoCamera(L, R, resized):
    camL = cv2.VideoCapture(int(L), cv2.CAP_DSHOW)
    camR = cv2.VideoCapture(int(R), cv2.CAP_DSHOW)

    if not camL.isOpened() & camR.isOpened():
        errorMessage("Cannot open webcam!")


    # Show original dimension
    widthL, heightL, widthR, heightR = originalDimCheck(camL, camR)

    print("\nVideo 1 original dimension: " + str(widthL) + ' ' + str(heightL))
    print("Video 2 original dimension: " + str(widthR) + ' ' + str(heightR))

    if widthL == resized[0] and heightL == resized[1] and widthR == resized[0] and heightR == resized[1]:
        print("\nINGFO: Two cameras have the same dimensions, no need to resized!")
    else:
        # Show resized dimension
        print("\nResized to: " + str(resized))

    print("\nSuccess: Stereo Camera successfully loaded!")

    return camL, camR

def stereoCalibrated():
    # Take camera parameters
    if not os.path.exists(os.getcwd() + '\\calibration\\stereoMap.xml'):
        errorMessage("'stereoMap.xml' is not found!")

    cv_file = cv2.FileStorage()
    cv_file.open(os.getcwd() + '\\calibration\\stereoMap.xml', cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y


def resizedStereoCamera(L, R, mapLx, mapLy, mapRx, mapRy, resize):
    retL, frameL = L.read()
    retR, frameR = R.read()

    if not frameL.shape[1] == resize[0] and frameL.shape[0] == resize[1]:
        # Resized input based from dimension
        frameL = cv2.resize(frameL, resize, interpolation=cv2.INTER_AREA)
        frameR = cv2.resize(frameR, resize, interpolation=cv2.INTER_AREA)

    # Remap frame based from stereoMap
    # frameL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    # frameR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Convert to grayscale
    frameGrayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    frameGrayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # If frame error then break
    if not retL & retR:
        errorMessage("Frame L/R got an error!")

    key = cv2.waitKey(1)

    return frameL, frameR, frameGrayL, frameGrayR, key

def destroySession(L, R):
    L.release()
    R.release()
    cv2.destroyAllWindows()
    print("\nThank you!\n:)")