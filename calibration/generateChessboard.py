# Take chessboard pictures for Calibration
# for ``unfold`` by Dandy Garda

# Import OpenCV, os and time
import cv2
import os
import time

print("\n== TAKE CHESSBOARD PICTURES FOR CALIBRATION ==\n``unfold`` by dandy garda\n")

# Input for how many times to take
totalShot = input("Masukkan berapa kali ambil gambar: ")

# Input Camera
cam1 = input("Masukkan nomor source left camera (1/2/3/..): ")
cam2 = input("Masukkan nomor source right camera (1/2/3/..): ")

# Resized Dimension
dim = (640, 480)

# For label name & total taken
index = 0

# Get unix timestamp for 1 session
getTime = str(int(time.time()))

# Check if input is not numeric
if not cam1.isnumeric() & cam2.isnumeric() & totalShot.isnumeric():
    print("\nInput unknown!")
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


# Left Camera
print("\nIt's time to left camera!\n")
while True:
    # Read webcam
    ret, frame = vid1.read()

    # If frame error then break
    if not ret:
        break

    cv2.imshow("Left Camera", frame)
    
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

        # Save
        cv2.imwrite(os.getcwd() + '\\calibration\\images\\' + getTime + '\\left\\' + str(index) + '.jpg', frame)
        print("Saved in " + os.getcwd() + '\\calibration\\images\\' + getTime + '\\left\\' + str(index) + '.jpg')

        index += 1
    
    if index == int(totalShot):
        index = 0
        break

    # Quit button
    if key == ord('q'):
        vid1.release()
        cv2.destroyAllWindows()
        quit()



vid1.release()
cv2.destroyAllWindows()



# Right Camera
print("\n\nIt's time to Right camera!\n")
while True:
    # Read webcam
    ret, frame = vid2.read()

    # If frame error then break
    if not ret:
        break

    cv2.imshow("Right Camera", frame)
    
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

        # Save
        cv2.imwrite(os.getcwd() + '\\calibration\\images\\' + getTime + '\\right\\' + str(index) + '.jpg', frame)
        print("Saved in " + os.getcwd() + '\\calibration\\images\\' + getTime + '\\right\\' + str(index) + '.jpg')

        index += 1
    
    if index == int(totalShot):
        index = 0
        break

    # Quit button
    if key == ord('q'):
        break

vid2.release()
cv2.destroyAllWindows()

print("Done, semoga berhasil!")