import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import OrderedDict

cap = cv2.VideoCapture(0)
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
MIN_MATCH_COUNT = 10
img1 = cv2.imread('object1.png') # trainImage
img2 = cv2.imread('object2.png')
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        img0 = frame
        kp0, des0 = sift.detectAndCompute(img0,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches1 = flann.knnMatch(des0,des1,k=2)
        matches2 = flann.knnMatch(des0,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        #lengths = []
        good = []

        good1 = []
        for m,n in matches1:
            if m.distance < 0.7*n.distance:
                good1.append(m)

        good2 = []
        for m,n in matches2:
            if m.distance < 0.7*n.distance:
                good2.append(m)

        a_dictionary = {"object1": len(good1), "object2": len(good2)}

        max_key = max(a_dictionary, key=a_dictionary.get)

        if a_dictionary[max_key]>MIN_MATCH_COUNT:
            output = max_key
        else:
            output = "No matches found"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img0,output, (10,45), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Display the resulting frame
        cv2.imshow('Frame',img0)
    #Pess Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
            # Break the loop
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
