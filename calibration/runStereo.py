# Run stereo camera based from stereoMap.xml to undistort and rectify camera in realtime
# for ``unfold`` by dandy garda

from re import L
import cv2
import os

# Take camera parameters
cv_file = cv2.FileStorage()
cv_file.open(os.getcwd() + '\\calibration\\stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Open cameras
print("\n== APPLY & RUN STEREO CAMERA ==\n``unfold`` by dandy garda\n")

print("Path stereo map: " + os.getcwd() + '\\calibration\\stereoMap.xml')

inputL = input("Masukkan input kamera kiri (1, 2, 3, ...): ")
inputR = input("Masukkan input kamera kanan (1, 2, 3, ...): ")

if not inputL.isnumeric() & inputR.isnumeric():
    print("\nInput unknown!")
    quit()

camL = cv2.VideoCapture(int(inputL), cv2.CAP_DSHOW)
camR = cv2.VideoCapture(int(inputR), cv2.CAP_DSHOW)

# Error if webcam is can't be opened
if not camL.isOpened() & camR.isOpened():
    raise IOError("\nCannot open webcam!")

# Print the exit
print("\nPress 'q' to exit!")

while True:
    retL, frameL = camL.read()
    retR, frameR = camR.read()

    # Undistort and rectify images in realtime
    frameL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frameR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Right Camera", frameR)

    key = cv2.waitKey(1)

    # Exit key with 'q'
    if key == ord('q') or key == ord('Q'):
        break

camL.release()
camR.release()
cv2.destroyAllWindows()