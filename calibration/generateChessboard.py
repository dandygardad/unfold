# Take chessboard pictures for Calibration
# for ``unfold`` by Dandy Garda

# Import OpenCV, os and time
import cv2
import os
import time

print("\n== TAKE CHESSBOARD PICTURES FOR CALIBRATION ==\n``unfold`` by dandy garda\n")

# Input Camera
cam1 = input("Masukkan nomor source left camera (1/2/3/..): ")
cam2 = input("Masukkan nomor source right camera (1/2/3/..): ")

# Resized Dimension
dim = (640, 480)

# For label name
index = 0

# Get unix timestamp for 1 session
getTime = str(int(time.time()))

# Check if input is not numeric
if not cam1.isnumeric() & cam2.isnumeric():
    print("Input unknown!")
    quit()

# Capture camera
vid1 = cv2.VideoCapture(int(cam1), cv2.CAP_DSHOW)
vid2 = cv2.VideoCapture(int(cam2), cv2.CAP_DSHOW)

# Error if webcam is can't be opened
if not vid1.isOpened() & vid2.isOpened():
    raise IOError("Cannot open webcam")

# Show original dimension
print("\nVideo 1 original dimension: " + str(vid1.get(cv2.CAP_PROP_FRAME_WIDTH)) + ' ' + str(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("\nVideo 2 original dimension: " + str(vid2.get(cv2.CAP_PROP_FRAME_WIDTH)) + ' ' + str(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Show resized dimension
print("\nResized to: " + str(dim))

print("\n\nPress 's' to take a picture left and right\nPress 'q' to quit")



while True:
    # Read webcam
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()

    # Resized input based from dimension
    resized1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

    # If frame error then break
    if not ret1 & ret2:
        break

    cv2.imshow("Left Camera", resized1)
    cv2.imshow("Right Camera", resized2)

    key = cv2.waitKey(1)

    # Take a pictuh with key 's'
    if key == ord('s'):
        # Make folder for storing images if not exists
        if not os.path.exists(os.getcwd() + '\\calibration\\images\\'):
            os.makedirs(os.getcwd() + '\\calibration\\images\\')

        # Make folder based from unix
        if not os.path.exists(os.getcwd() + '\\calibration\\images\\' + getTime):
            os.makedirs(os.getcwd() + '\\calibration\\images\\' + getTime + '\\left\\')
            os.makedirs(os.getcwd() + '\\calibration\\images\\' + getTime + '\\right\\')

        # Left Camera
        cv2.imwrite(os.getcwd() + '\\calibration\\images\\' + getTime + '\\left\\' + str(index) + '.jpg', resized1)
        # Right Camera
        cv2.imwrite(os.getcwd() + '\\calibration\\images\\' + getTime + '\\right\\' + str(index) + '.jpg', resized2)
        
        print("Saved in " + os.getcwd() + '\\calibration\\images\\' + getTime + '\\left\\' + str(index) + '.jpg')
        print("Saved in " + os.getcwd() + '\\calibration\\images\\' + getTime + '\\right\\' + str(index) + '.jpg')
        
        index += 1

    # Quit button
    if key == ord('q'):
        break

vid1.release()
vid2.release()
cv2.destroyAllWindows()