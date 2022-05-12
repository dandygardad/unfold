# Main application
# for ``unfold`` by dandy garda

# Import libraries
import cv2
import torch
import numpy as np

from helper.general import unfoldHeader, errorMessage
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated



# ``unfold`` Header
unfoldHeader()



# LOAD STEREO CAMERA
print("=== LOAD STEREO CAMERA ===")
inputL = input("Masukkan input kamera kiri (0/1/2/3/..): ")
inputR = input("Masukkan input kamera kanan (0/1/2/3/..): ")

if not inputL.isnumeric() & inputR.isnumeric():
    errorMessage("Input source camera is not numeric!")

dim = (640, 480)
camL, camR = stereoCamera(inputL, inputR, dim)



# LOAD MODEL (YOLOv5)
print("\n\n=== RUNNING YOLOv5 ===")
try:
    model = torch.hub.load('yolov5-detect', 'custom', path='./models/yolov5s.pt', source='local')
except:
    errorMessage("Cannot load model, please check 'torch.hub.load' function!")


# RUN YOLOv5 TO OpenCV
print("\n\n=== PUT YOLOv5 INTO STEREO CAMERA ===")
stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = stereoCalibrated()
while True:
    try:
        # Load stereo camera
        resized1, resized2, key = resizedStereoCamera(camL, camR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, dim)
        resultLR = model([resized1[:, :, ::-1], resized2[:, :, ::-1]])

        # Print into command prompt
        resultLR.print()

        # Show realtime
        cv2.imshow("Left Camera", resultLR.render()[0][:, :, ::-1])
        cv2.imshow("Right Camera", resultLR.render()[1][:, :, ::-1])

        # Key to exit
        if key == ord('q'):
            print("\nExited!")
            break
        
        print()
    except KeyboardInterrupt:
        print("\nExited!")
        break

# Destroy Session
destroySession(camL, camR)

