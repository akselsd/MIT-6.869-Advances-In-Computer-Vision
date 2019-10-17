# OPTIONAL: tool for interactively selecting keypoints for images
# mouse click for selecting keypoints in one direction along the contour
# when finished, press any key to exit and save the keypoints

import cv2
import scipy.io
import numpy as np

X = []
Y = []

# mouse callback, store the keypoints when mouse clicked
def click_keypoints(event, x, y, flags, param):
    # grab references to the global variables
    global X, Y

    # mouse clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        X.append([x+1]) # index + 1 for matlab consistency
        Y.append([y+1])
        print(x, y)
 
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
        cv2.imshow("frame", img)

# set mouse callback
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click_keypoints)

# load image
img = cv2.imread('flower.jpg')

# display image
cv2.imshow('frame', img)
cv2.waitKey(0)

# save keypoints
scipy.io.savemat('output.mat', {
    'x': np.array(X),
    'y': np.array(Y)
}) 