# Main application
# for ``unfold`` by dandy garda

# Import libraries
import pandas as pd
import cv2
import torch
import json

from helper.general import unfoldHeader, errorMessage, errorDetection
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated
from helper.distance import convertBbox, stereoscopicMeasurement, bboxLabelDistance



# ``unfold`` Header
unfoldHeader()


# GET DATA FROM JSON
f = open('changeData.json')
dataJson = json.load(f)
f.close()

# Load data from json
if dataJson['cameraConfig']['model']:
    # Custom model
    model_custom = './models/' + dataJson['cameraConfig']['model']
else:
    # Default model by YOLOv5 (Coco128)
    model_custom = './models/' + 'yolov5s.pt'

if dataJson['cameraConfig']['conf']:
    conf_custom = dataJson['cameraConfig']['conf']
else:
    conf_custom = 0

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
    model = torch.hub.load('yolov5-detect', 'custom', path=model_custom, source='local')
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
        ###### STEREO CAMERA & SETTINGS ######

        # Load stereo camera
        resized1, resized2, resizedGrayL, resizedGrayR, key = resizedStereoCamera(camL, camR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, dim)
        
        # Inference Settings
        # EDIT THIS FOR DETECTION SETTINGS
        model.conf = conf_custom

        if dataJson['cameraConfig']['customModel'] != False:
            model.classes = dataJson['cameraConfig']['customModel']
            

        # Load frame to model
        resultLR = model([resizedGrayL, resizedGrayR], augment=True)

        ###### END OF STEREO CAMERA & SETTINGS ######

        
        ###### PRINT INTO COMMAND PROMPT ######

        labelL = resultLR.pandas().xyxy[0] # (Left Camera)
        labelR = resultLR.pandas().xyxy[1]  # (Right Camera)

        print("\n--------------------------------------------\n")
        resultLR.print()

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
                        distance = stereoscopicMeasurement(xl, xr, dim[0], dataJson['cameraConfig']['baseline'], dataJson['cameraConfig']['fieldOfView'])

                        classes.append(labelL.iloc[id]['name'])
                        distances.append(distance)
                    else:
                        resultImgL, resultImgR = errorDetection("Class Left & Right is not same!", resized1, resized2)
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

                    # Put manual bbox and distance in frame
                    resultImgL = bboxLabelDistance(labelL, data, resized1)
                    resultImgR = bboxLabelDistance(labelR, data, resized2)
            else:
                resultImgL, resultImgR = errorDetection("Total label in L doesn't same as total label in R", resized1, resized2)
        else:
            resultImgL, resultImgR = errorDetection("No detection on left/right camera!", resized1, resized2)

        ###### END OF PRINT TO COMMAND ######

        # Show camera in realtime
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

