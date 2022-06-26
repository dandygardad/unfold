# Functions to help stereo camera running and calibrated
# for ``unfold`` by dandy garda

import cv2
import os
from helper.general import originalDimCheck, errorMessage

# Load Stereo Camera
def stereoCamera(L, R, dshow):
    if dshow:
        camL = cv2.VideoCapture(L, cv2.CAP_DSHOW)
        camR = cv2.VideoCapture(R, cv2.CAP_DSHOW)
    else:
        camL = cv2.VideoCapture(L)
        camR = cv2.VideoCapture(R)

    if not camL.isOpened() & camR.isOpened():
        errorMessage("Cannot open webcam!")

    # Show original dimension
    widthL, heightL, widthR, heightR = originalDimCheck(camL, camR)

    print("\nVideo 1 original dimension: " + str(widthL) + ' ' + str(heightL))
    print("Video 2 original dimension: " + str(widthR) + ' ' + str(heightR))

    print("\nSuccess: Stereo Camera successfully loaded!")

    return camL, camR, widthL, heightL, widthR, heightR

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

    if retL & retR:
        # Remap frame based from stereoMap
        frameL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        frameR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        
        # Convert to grayscale
        frameGrayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        frameGrayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    else:
        return retL, retR, False, False, False, False, False


    key = cv2.waitKey(1)

    return retL, retR, frameL, frameR, frameGrayL, frameGrayR, key

def destroySession(L, R):
    L.release()
    R.release()
    cv2.destroyAllWindows()
    print("\nThank you!\n:)")