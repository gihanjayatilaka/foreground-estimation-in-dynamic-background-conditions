
import cv2 as cv
import numpy as np
from PIL import Image



img = cv.imread("img7.jpg")

OLD_H=img.shape[0]
OLD_W=img.shape[1]

NEW_H=int(OLD_H/4)
NEW_W=int(OLD_W/4)

newImg=np.ndarray(shape=(NEW_H,NEW_W),dtype=np.uint8)

for r in range(NEW_H):
    for c in range(NEW_W):
        greyVal=0
        for rr in range(4):
            for cc in range(4):
                for z in range(3):
                    greyVal+=img[r*4+rr][c*4+cc][z]
        newImg[r][c]=int(greyVal/(4*4*3))

print(newImg)
newImg.astype(np.uint8)

cv.imwrite("img7.tiff",newImg)
cv.imshow("Resized",newImg)
cv.waitKey(3000) & 0xff