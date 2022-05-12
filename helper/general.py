import cv2
import os

# Header template
def unfoldHeader():
    os.system('cls')
    print('\033[92m``unfold``\033[0m v.1.0.0')
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
    print("\nVideo 1 original dimension: " + str(L.get(cv2.CAP_PROP_FRAME_WIDTH)) + ' ' + str(L.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Video 2 original dimension: " + str(R.get(cv2.CAP_PROP_FRAME_WIDTH)) + ' ' + str(R.get(cv2.CAP_PROP_FRAME_HEIGHT)))