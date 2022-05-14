# Stereoscopic Measurement
# for ``unfold`` by dandy garda

import math

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
    baselineWidth = b * width
    disparity = leftX - rightX
    fieldOfView = math.tan(fov / 2)

    distance = baselineWidth / (2 * fieldOfView * disparity)

    return distance