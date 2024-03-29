# General functions
# for ``unfold`` by dandy garda

import cv2
import os

# Header template
def unfoldHeader(cls):
    if cls:
        os.system('cls')
    else:
        print('\n')
    print('\033[92m``unfold``\033[0m v.2.1.0')
    print('\033[1mA Python project for measuring distance between two ships with Stereo Camera.\033[0m')
    print('\n\033[1mmade by Dandy Garda\033[0m')
    print('\033[1mgowa, 2022\033[0m')
    print('\n-----------------------------------------------------------------------------\n')


# Template for error message
def errorMessage(msg):
    print("\n\033[91mERRRRRR!!\033[0m")
    print("Message: " + msg)
    quit()

# Check original dimension
def originalDimCheck(L, R):
    if L.get(cv2.CAP_PROP_FRAME_WIDTH) != R.get(cv2.CAP_PROP_FRAME_WIDTH) and L.get(cv2.CAP_PROP_FRAME_HEIGHT) != R.get(cv2.CAP_PROP_FRAME_HEIGHT):
        errorDetection("Dimensions from 2 camera are not same!", L, R)
        quit()

    return L.get(cv2.CAP_PROP_FRAME_WIDTH), L.get(cv2.CAP_PROP_FRAME_HEIGHT), R.get(cv2.CAP_PROP_FRAME_WIDTH), R.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Template error related from detection
def errorDetection(msg, frameL, frameR):
    print("\nERRRR: " + msg)
    return frameL, frameR