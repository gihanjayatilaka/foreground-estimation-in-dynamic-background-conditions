
import cv2 as cv
import numpy as np
from PIL import Image



img = cv.imread("img4.png")

OLD_H=img.shape[0]
OLD_W=img.shape[1]

NEW_H=(OLD_H)
NEW_W=(OLD_W)

newImg=np.ndarray(shape=(NEW_H,NEW_W),dtype=np.uint8)

for r in range(NEW_H):
    for c in range(NEW_W):
        greyVal=0
        for z in range(3):
            greyVal+=img[r][c][z]
        newImg[r][c]=int(greyVal/(3))

print(newImg)
newImg.astype(np.uint8)


cv.imwrite("img4.tiff",newImg)
cv.imshow("Resized",newImg)
cv.waitKey(3000) & 0xff