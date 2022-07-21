# Main application
# for ``unfold`` by dandy garda

# Import libraries and functions
import pandas as pd
import cv2
import torch
import yaml
import os

from helper.general import unfoldHeader, errorMessage, errorDetection
from helper.load import resizedStereoCamera, stereoCamera, destroySession, stereoCalibrated
from helper.distance import convertBbox, stereoscopicMeasurement, bboxLabelDistance
from helper.rmse import saveData







###### LOAD YAML ######

f = open('config.yaml')
dataConfig = yaml.safe_load(f)
f.close()

# ``unfold`` Header
unfoldHeader(dataConfig['header']['cls'])

# Load data from yaml
if dataConfig['cameraConfig']['model']:
    # Custom model
    model_custom = './models/' + dataConfig['cameraConfig']['model']
else:
    # Default model by YOLOv5 (Coco128)
    model_custom = './models/' + 'yolov5s.pt'

if dataConfig['cameraConfig']['conf']:
    conf_custom = dataConfig['cameraConfig']['conf']
else:
    conf_custom = 0

if dataConfig['rmse']['mode']:
    mode_rmse = dataConfig['rmse']['mode']
    dist_rmse = dataConfig['rmse']['setDistance']
    frame_rmse = dataConfig['rmse']['maxFramesPerDist']
    result_rmse = {}
    distances_rmse = list()

    print("INGFO: Mode RMSE ON!\n")
else:
    mode_rmse = False

mode_capture = dataConfig['capture']['mode']

if mode_capture == 'video' and dataConfig['capture']['cam1'] and dataConfig['capture']['cam2']:
    cam1_capture = dataConfig['capture']['cam1']
    cam2_capture = dataConfig['capture']['cam2']
elif mode_capture == 'video':
    _, _ = errorMessage("Cam 1/Cam 2 source video contains false!")
    quit()


###### END OF LOAD YAML ######




###### LOAD STEREO CAMERA ######

if mode_capture == 'video':
    print("=== LOAD VIDEO ===")
    inputL = os.getcwd() + '\\video\\' + cam1_capture
    inputR = os.getcwd() + '\\video\\' + cam2_capture
    camL, camR, widthL, heightL, widthR, heightR = stereoCamera(inputL, inputR, False)
else:
    print("=== LOAD STEREO CAMERA ===")
    inputL = input("Masukkan input kamera kiri (0/1/2/3/..): ")
    inputR = input("Masukkan input kamera kanan (0/1/2/3/..): ")

    if not inputL.isnumeric() & inputR.isnumeric():
        errorMessage("Input source camera is not numeric!")

    camL, camR, widthL, heightL, widthR, heightR = stereoCamera(int(inputL), int(inputR), True)
    
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
        retL, retR, resized1, resized2, resizedGrayL, resizedGrayR, key = resizedStereoCamera(camL, camR, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, dim)
        
        if not retL & retR:
            break

        # Inference Settings
        model.conf = conf_custom

        if dataConfig['cameraConfig']['customModel'] != False:
            model.classes = dataConfig['cameraConfig']['customModel']
            

        # Load frame to model
        resultLR = model([resizedGrayL], augment=True)

        ###### END OF STEREO CAMERA & SETTINGS ######




        ###### MATCH TEMPLATE ######

        labelL = resultLR.pandas().xyxy[0] # (Left Camera)
        labelR = pd.DataFrame({})

        for i in range(len(labelL)):
            image = resizedGrayL[int(labelL.iloc[i]['ymin']):int(labelL.iloc[i]['ymax']), int(labelL.iloc[i]['xmin']):int(labelL.iloc[i]['xmax'])]
            height, width = image.shape[::]
            match = cv2.matchTemplate(resizedGrayR, image, cv2.TM_SQDIFF)
            _, _, minloc, maxloc = cv2.minMaxLoc(match)
            data = {
                "xmin": float(minloc[0]),
                "ymin": float(minloc[1]),
                "xmax": float(minloc[0] + width),
                "ymax": float(minloc[1] + height),
                "confidence": labelL.iloc[i]['confidence'],
                "class": labelL.iloc[i]['class'],
                "name": labelL.iloc[i]['name']
            }
            labelR = pd.concat([labelR, pd.DataFrame(data, index=[i])]) 
            
        ###### END OF MATCH TEMPLATE ######




        ###### PRINT INTO COMMAND PROMPT ######

        print("\n--------------------------------------------")

        if len(labelL) and len(labelR):
            labelR = labelR.sort_values(by=['confidence'], ascending=False)  
            if len(labelL) == len(labelR):
                
                print("\nDetection on Left Camera: ")
                print(labelL)
                print("\nDetection on Right Camera (from template matching): ")
                print(labelR)

                id = 0
                while id < len(labelL):
                    # Converting float into int for stability value
                    xl, yl, wl, hl = convertBbox(round(labelL.iloc[id]['xmin'], dataConfig['cameraConfig']['detectRound']), round(labelL.iloc[id]['ymin'], dataConfig['cameraConfig']['detectRound']), round(labelL.iloc[id]['xmax'], dataConfig['cameraConfig']['detectRound']), round(labelL.iloc[id]['ymax'], dataConfig['cameraConfig']['detectRound']))
                    xr, yr, wr, hr = convertBbox(labelR.iloc[id]['xmin'], labelR.iloc[id]['ymin'], labelR.iloc[id]['xmax'], labelR.iloc[id]['ymax'])

                    if dataConfig['cameraConfig']['blockDiffClass']:
                        # If two class from cameras are not same then break
                        if labelL.iloc[id]['name'] == labelR.iloc[id]['name']:
                            print("\n\nx1 for left camera = " + str(xl))
                            print("x2 for right camera = " + str(xr))

                            # Result from Distance Measurement
                            distance = stereoscopicMeasurement(xl, xr, dim[0], dataConfig['cameraConfig']['baseline'], dataConfig['cameraConfig']['fieldOfView'])
                            
                            classes.append(labelL.iloc[id]['name'])
                            distances.append(distance)

                            # Append distance into RMSE
                            if mode_rmse:
                                if not labelL.iloc[id]['name'] in result_rmse:
                                    result_rmse[labelL.iloc[id]['name']] = list()
                                result_rmse[labelL.iloc[id]['name']].append(round(distance, dataConfig['rmse']['distRound']))
                        else:
                            resultImgL, resultImgR = errorDetection("Class Left & Right is not same!", resized1, resized2)
                            break
                    else:
                        print("\n\nx1 for left camera = " + str(xl))
                        print("x2 for right camera = " + str(xr))

                        # Result from Distance Measurement
                        distance = stereoscopicMeasurement(xl, xr, dim[0], dataConfig['cameraConfig']['baseline'], dataConfig['cameraConfig']['fieldOfView'])

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

        if dataConfig['cameraConfig']['combinedCamera']:
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

