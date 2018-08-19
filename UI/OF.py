import cv2
import numpy as np

def OF(prev, nxt, hsv):
    prev = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(nxt,cv2.COLOR_BGR2GRAY)
#    flow = cv2.calcOpticalFlowFarneback(prev,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(prev,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #print(flow.shape)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #print(mag.shape, ang.shape)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr
