# Main application
# for ``unfold`` by dandy garda

# Import libraries and functions
import pandas as pd
import cv2
import torch
import json

from helper.general import unfoldHeader, errorMessage, errorDetection
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated
from helper.distance import convertBbox, stereoscopicMeasurement, bboxLabelDistance
from helper.rmse import saveData







###### LOAD JSON ######

f = open('changeData.json')
dataJson = json.load(f)
f.close()

# ``unfold`` Header
unfoldHeader(dataJson['header']['cls'])

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

if dataJson['rmse']['mode']:
    mode_rmse = dataJson['rmse']['mode']
    dist_rmse = dataJson['rmse']['setDistance']
    frame_rmse = dataJson['rmse']['maxFramesPerDist']
    result_rmse = {}
    distances_rmse = list()

    print("INGFO: Mode RMSE ON!\n")
else:
    mode_rmse = False

###### END OF LOAD JSON ######




###### LOAD STEREO CAMERA ######

print("=== LOAD STEREO CAMERA ===")
inputL = input("Masukkan input kamera kiri (0/1/2/3/..): ")
inputR = input("Masukkan input kamera kanan (0/1/2/3/..): ")

if not inputL.isnumeric() & inputR.isnumeric():
    errorMessage("Input source camera is not numeric!")

camL, camR, widthL, heightL, widthR, heightR = stereoCamera(inputL, inputR)
# Assume two cameras are same model
dim = (widthL, heightL)

###### END OF LOAD STEREO CAMERA ######




###### LOAD YOLOv5 ######

print("\n\n=== RUNNING YOLOv5 ===")
try:
    model = torch.hub.load('yolov5-detect', 'custom', path=model_custom, source='local')
except Exception as e:
    print(e)
    errorMessage("Cannot load model, please check 'torch.hub.load' function!")

###### END OF LOAD YOLOv5 ######




###### RUN YOLOv5 TO OpenCV ######

print("\n\n=== PUT YOLOv5 INTO STEREO CAMERA ===")
print("=== APPLY DISTANCE MEASUREMENT ===")
stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = stereoCalibrated()
initial_frame = 0

while True:
    if mode_rmse:
        print("\nFrame: " + str(initial_frame))
        if initial_frame == frame_rmse:
            saveData(dist_rmse, result_rmse)
            break

    classes = list()
    distances = list()
    try:
        ###### STEREO CAMERA & SETTINGS ######

        # Load stereo camera
        resized1, resized2, resizedGrayL, resizedGrayR, key = resizedStereoCamera(camL, camR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, dim)
        
        # Inference Settings
        model.conf = conf_custom

        if dataJson['cameraConfig']['customModel'] != False:
            model.classes = dataJson['cameraConfig']['customModel']
            

        # Load frame to model
        resultLR = model([resizedGrayL, resizedGrayR], augment=True)

        ###### END OF STEREO CAMERA & SETTINGS ######

        


        ###### PRINT INTO COMMAND PROMPT ######

        labelL = resultLR.pandas().xyxy[0] # (Left Camera)
        labelR = resultLR.pandas().xyxy[1]  # (Right Camera)

        print("\n--------------------------------------------")

        if len(labelL) and len(labelR):
            if len(labelL) == len(labelR):
                # Add index into result name
                for i in range(len(labelR)):
                    labelL.at[i, 'name'] = labelL.iloc[i]['name'] + str(i)
                    labelR.at[i, 'name'] = labelR.iloc[i]['name'] + str(i)
                
                print("\nDetection on Left Camera: ")
                print(labelL)
                print("\nDetection on Right Camera: ")
                print(labelR)

                id = 0
                while id < len(labelL):
                    xl, yl, wl, hl = convertBbox(labelL.iloc[id]['xmin'], labelL.iloc[id]['ymin'], labelL.iloc[id]['xmax'], labelL.iloc[id]['ymax'])
                    xr, yr, wr, hr = convertBbox(labelR.iloc[id]['xmin'], labelR.iloc[id]['ymin'], labelR.iloc[id]['xmax'], labelR.iloc[id]['ymax'])

                    if dataJson['cameraConfig']['blockDiffClass']:
                        # If two class from cameras are not same then break
                        if labelL.iloc[id]['name'] == labelR.iloc[id]['name']:
                            print("\n\nx1 for left camera = " + str(xl))
                            print("x2 for right camera = " + str(xr))

                            # Result from Distance Measurement
                            distance = stereoscopicMeasurement(xl, xr, dim[0], dataJson['cameraConfig']['baseline'], dataJson['cameraConfig']['fieldOfView'])
                            classes.append(labelL.iloc[id]['name'])
                            distances.append(distance)

                            # Append distance into RMSE
                            if mode_rmse:
                                if not labelL.iloc[id]['name'] in result_rmse:
                                    result_rmse[labelL.iloc[id]['name']] = list()
                                result_rmse[labelL.iloc[id]['name']].append(distance)
                        else:
                            resultImgL, resultImgR = errorDetection("Class Left & Right is not same!", resized1, resized2)
                            break
                    else:
                        print("\n\nx1 for left camera = " + str(xl))
                        print("x2 for right camera = " + str(xr))

                        # Result from Distance Measurement
                        distance = stereoscopicMeasurement(xl, xr, dim[0], dataJson['cameraConfig']['baseline'], dataJson['cameraConfig']['fieldOfView'])

                        classes.append(labelL.iloc[id]['name'])
                        distances.append(distance)

                        # Append distance into RMSE
                        if mode_rmse:
                            if not labelL.iloc[id]['name'] in result_rmse:
                                result_rmse[labelL.iloc[id]['name']] = list()
                            result_rmse[labelL.iloc[id]['name']].append(distance)

                    id += 1
                initial_frame += 1

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




        ###### SHOW CAMERAS IN REALTIME ######

        if dataJson['cameraConfig']['combinedCamera']:
            # Combine two frame into one
            alpha = 0.5
            beta = (1.0 - alpha)
            combineImg = cv2.addWeighted(resultImgR, alpha, resultImgL, beta, 0.0)
            cv2.imshow("Combined Cameras", combineImg)
        else:
            cv2.imshow("Left Camera", resultImgL)
            cv2.imshow("Right Camera", resultImgR)

        ###### END OF SHOW CAMERAS IN REALTIME ######

        

        # Key to exit
        if key == ord('q') or key == ord('Q'):
            if mode_rmse:
                saveData(dist_rmse, result_rmse)
            print("\n\nExited!")
            break

    except KeyboardInterrupt:
        if mode_rmse:
            saveData(dist_rmse, result_rmse)
        print("\n\nExited!")
        break

###### END OF RUN YOLOv5 TO OpenCV ######



# Destroy Session
destroySession(camL, camR)

