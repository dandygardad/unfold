# Main application
# for ``unfold`` by dandy garda

# Import libraries
import pandas as pd
import cv2
import torch

from helper.general import unfoldHeader, errorMessage
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated
from helper.distance import convertBbox, stereoscopicMeasurementV1



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

# classes = list()
# distance = list()
# x = list()
# y = list()
# w = list()
# h = list()


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
        model.classes = [67]

        # Load frame to model
        resultLR = model([resized1[:, :, ::-1], resized2[:, :, ::-1]])

        # Show realtime
        cv2.imshow("Left Camera", resultLR.render()[0][:, :, ::-1])
        cv2.imshow("Right Camera", resultLR.render()[1][:, :, ::-1])

        # Key to exit
        if key == ord('q'):
            print("\n\nExited!")
            break

        # Print into command prompt
        print("\n--------------------------------------------\n")
        resultLR.print()

        # Extract bbox (x1, y1, x2, y2) to (x, y, w, h)
        try:
            labelL = resultLR.pandas().xyxy[0] # (Left Camera)
            labelR = resultLR.pandas().xyxy[1]  # (Right Camera)
            xl, yl, wl, hl = convertBbox(labelL.iloc[0]['xmin'], labelL.iloc[0]['ymin'], labelL.iloc[0]['xmax'], labelL.iloc[0]['ymax'])
            xr, yr, wr, hr = convertBbox(labelR.iloc[0]['xmin'], labelR.iloc[0]['ymin'], labelR.iloc[0]['xmax'], labelR.iloc[0]['ymax'])
            
            print("\n\nDetection on Left Camera: ")
            print(labelL)
            print("\nDetection on Right Camera: ")
            print(labelR)
            
            # At this time, only one class with highest conf can be measured
            if labelL.iloc[0]['name'] == labelR.iloc[0]['name']:
                # print("(" + label.iloc[0]['name'] + ')')
                print("\n\nx1 for left camera = " + str(xl))
                print("x2 for right camera = " + str(xr))

                # Result from Distance Measurement
                # CHANGE THIS IF THERE IS CHANGES ON BASELINE AND FOV
                distance = stereoscopicMeasurementV1(xl, xr, dim[0], 0.7, 170)
                
                data = {
                    'class': [labelL.iloc[0]['name']],
                    'distance': [distance]
                }

                print("\nDistance Measurement:")
                print(pd.DataFrame(data))
                
            else:
                print("Can't measure the distance!")    
        except IndexError:
            print("No detection!")
            continue
    except KeyboardInterrupt:
        print("\n\nExited!")
        break

# Destroy Session
destroySession(camL, camR)

