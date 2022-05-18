# Stereoscopic Measurement
# for ``unfold`` by dandy garda

import math
import cv2

# Create bbox manually
def bboxResult(dataBbox, frame):
    if len(dataBbox):
        i = 0
        while i < len(dataBbox):
            xmin = int(dataBbox.iloc[i]['xmin'])
            ymin = int(dataBbox.iloc[i]['ymin'])
            xmax = int(dataBbox.iloc[i]['xmax'])
            ymax = int(dataBbox.iloc[i]['ymax'])
            label = dataBbox.iloc[i]['name']

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            resultImg = cv2.rectangle(frame, (xmin, ymin), (xmin + text_w, ymin - text_h), (0, 0, 0), -1)

            resultImg = cv2.putText(frame, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            resultImg = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            i += 1

    else:
        return frame, False

    return resultImg, True

# Converting the results from PyTorch hub
def convertBbox(x1, y1, x2, y2):
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    width = (x2 - x1)
    height = (y2 - y1)

    return xc, yc, width, height


# Stereoscopic Measurement
"""
PARAMETERS:

leftX = target coordinates in the x-axis for left camera (px)
rightX = target coordinates in the x-axis for right camera (px)
width = width taken from image dimension (px)
b = baseline (actual distance between two cameras) (m)
fov = field of view/lens view angle (two cameras must be of the same model)
"""
def stereoscopicMeasurementV1(leftX, rightX, width, b, fov):
    baselineWidth = float(b) * float(width)
    disparity = float(leftX) - float(rightX)
    fieldOfView = float(math.tan(fov / 2))

    distance = baselineWidth / ((2 * fieldOfView) * disparity)

    return round(distance)