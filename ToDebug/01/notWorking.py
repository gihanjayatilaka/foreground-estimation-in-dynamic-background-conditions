import numpy as np
import cv2

cap = cv2.VideoCapture(0)

#Get the output qulity of cam
while (cap.isOpened()== True):
    ret, frame = cap.read()
    if ret == True:
        m, n, o = frame.shape
        break

sum=np.ndarray(shape=(100,m,n,o),dtype=int)

for x in range(100):
    ret,frame=cap.read()
    sum[x]=frame.copy()
    print (frame.shape)
    print(sum[x].shape)
    cv2.imshow('ThisFrame',sum[x])
    k = cv2.waitKey(30) & 0xff


cap.release()
cv2.destroyAllWindows()
