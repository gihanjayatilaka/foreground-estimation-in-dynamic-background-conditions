
import numpy as np
import cv2

cap = cv2.VideoCapture(0)


ssum=np.ndarray(shape=(100,480, 848, 3),dtype=int)

for x in range(100):
    ret,frame=cap.read()
    ssum[x]=frame
    print(ssum[x])
    cv2.imshow('ThisFrame',ssum)
    k = cv2.waitKey(30) & 0xff


cap.release()
cv2.destroyAllWindows()
