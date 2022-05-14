# Distance Measurement (Stereoscopic Measurement)
# for ``unfold`` by dandy garda

# Converting the results from PyTorch hub
def convertBbox(x1, y1, x2, y2):
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    width = (x2 - x1)
    height = (y2 - y1)

    return xc, yc, width, height

# Extract the disparity (x1, x2)
# x1 is for left camera
# x2 is for right camera
def disparity(leftX, rightX):
    pass


def distanceMeasurementV1(x, b, fov):
    pass