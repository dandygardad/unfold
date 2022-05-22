# Main application
# for ``unfold`` by dandy garda

# Import libraries
import pandas as pd
import cv2
import torch

from helper.general import unfoldHeader, errorMessage
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated
from helper.distance import convertBbox, stereoscopicMeasurementV1, bboxResult, bboxLabelDistance



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
    classes = list()
    distances = list()
    try:
        # Load stereo camera
        resized1, resized2, key = resizedStereoCamera(camL, camR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, dim)
        
        # Inference Settings
        # EDIT THIS FOR DETECTION SETTINGS
        model.conf = 0.4
        # model.classes = [0]

        # Load frame to model
        resultLR = model([resized1[:, :, ::-1], resized2[:, :, ::-1]])

        labelL = resultLR.pandas().xyxy[0] # (Left Camera)
        labelR = resultLR.pandas().xyxy[1]  # (Right Camera)

        # Print into command prompt
        print("\n--------------------------------------------\n")
        resultLR.print()

        # Make rectangle manual by bbox
        resultImgL, getL = bboxResult(labelL, resized1)
        resultImgR, getR = bboxResult(labelR, resized2)
        
        if not getL:
            print("\nERRRR: No detection in Left Camera!")
        if not getR:
            print("\nERRRR: No detection in Right Camera!")

        if len(labelL) and len(labelR):
            print("\n\nDetection on Left Camera: ")
            print(labelL)
            print("\nDetection on Right Camera: ")
            print(labelR)

            if len(labelL) == len(labelR):
                id = 0
                while id < len(labelL):
                    xl, yl, wl, hl = convertBbox(labelL.iloc[id]['xmin'], labelL.iloc[id]['ymin'], labelL.iloc[id]['xmax'], labelL.iloc[id]['ymax'])
                    xr, yr, wr, hr = convertBbox(labelR.iloc[id]['xmin'], labelR.iloc[id]['ymin'], labelR.iloc[id]['xmax'], labelR.iloc[id]['ymax'])

                    if labelL.iloc[id]['name'] == labelR.iloc[id]['name']:
                        print("\n\nx1 for left camera = " + str(xl))
                        print("x2 for right camera = " + str(xr))

                        # Result from Distance Measurement
                        # CHANGE THIS IF THERE IS CHANGES ON BASELINE AND FOV
                        distance = stereoscopicMeasurementV1(xl, xr, dim[0], 10, 78)

                        classes.append(labelL.iloc[id]['name'])
                        distances.append(distance)
                    else:
                        print("\nERRRR: Class Left & Right is not same!")
                        break
                    id += 1

                if len(classes):
                    data = {
                        'class': classes,
                        'distance': distances
                    }
                    data = pd.DataFrame(data)
                    
                    print("\nDistance Measurement:")
                    print(data)

            else:
                print("\nERRRR: Total label in L doesn't same as total label in R")    
        else:
            print("\nERRRR: Can't detect!")

        bboxLabelDistance(labelL, data, resultImgL)
        bboxLabelDistance(labelR, data, resultImgR)

        # Show realtime
        cv2.imshow("Left Camera", resultImgL)
        cv2.imshow("Right Camera", resultImgR)


        # Key to exit
        if key == ord('q'):
            print("\n\nExited!")
            break

    except KeyboardInterrupt:
        print("\n\nExited!")
        break

# Destroy Session
destroySession(camL, camR)

