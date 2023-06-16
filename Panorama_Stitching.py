from IPython.display import Image
import skimage
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import imutils

def panorama_stitching (img1, img2):
    stitcher = cv2.Stitcher.create()
    result = stitcher.stitch((img1, img2))
    stitched = result[1]

    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    mask = np.zeros(thresh.shape,dtype="uint8")
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect,None)
        sub = cv2.subtract(minRect,thresh)

    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    stitched = stitched[y:y+h, x:x+w]

    cv2.imwrite('scottsdale_stitched.png', stitched)
    img_end = cv2.cvtColor(stitched,cv2.COLOR_BGR2RGB)

    return img_end

img1 = cv2.imread('./panorama/scottsdale_left.png')
img2 = cv2.imread('./panorama/scottsdale_right.png')

img_match = panorama_stitching(img1,img2)
plt.figure(figsize=(10, 10))
plt.imshow(img_match)
plt.show()

