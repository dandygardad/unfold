# Main application
# for ``unfold`` by dandy garda

# Import libraries
import cv2
import torch
import numpy as np

from helper.general import unfoldHeader, errorMessage
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated
from helper.distance import convertBbox



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
except Exception as e:
    print(e)
    errorMessage("Cannot load model, please check 'torch.hub.load' function!")


# RUN YOLOv5 TO OpenCV
print("\n\n=== PUT YOLOv5 INTO STEREO CAMERA ===")
print("=== APPLY DISTANCE MEASUREMENT ===")
stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = stereoCalibrated()
while True:
    try:
        # Load stereo camera
        resized1, resized2, key = resizedStereoCamera(camL, camR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, dim)
        
        # Inference Settings
        # EDIT THIS FOR DETECTION SETTINGS
        model.conf = 0.4
        model.classes = [0]

        # Load frame to model
        resultLR = model([resized1[:, :, ::-1], resized2[:, :, ::-1]])

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

    # Print into command prompt
    resultLR.print()

    # Extract bbox (x1, y1, x2, y2) to (x, y, w, h)
    try:
        # label = resultLR.pandas().xyxy[0] # (Left Camera)
        label = resultLR.pandas().xyxy[1]  # (Right Camera)
        x, y, w, h = convertBbox(label.iloc[0]['xmin'], label.iloc[0]['ymin'], label.iloc[0]['xmax'], label.iloc[0]['ymax'])
        print(label)
        print("(" + label.iloc[0]['name'] + ')')
        print("x = " + str(x))
        print("y = " + str(y))
        print("w = " + str(w))
        print("h = " + str(h))
    except IndexError:
        print("No detection!")
        continue

# Destroy Session
destroySession(camL, camR)

