import numpy as np
import cv2

cap = cv2.VideoCapture(0)
_, frame = cap.read()

N = 30
N_min = 2


width, height = frame.shape[0], frame.shape[1]

AVG = np.zeros((width, height, 3))
FG_MASK = np.zeros((width, height))
BG_MODEL = []
R = np.zeros((width, height))
R+=100
R_scale =5
R_inc_dec = 0.01
T = np.zeros((width,height)) # not added yet
DISTMIN = np.zeros((width, height))


for i in range(100):
    _, frame = cap.read()
# first few frames
for i in range(N):
    _, frame = cap.read()
    BG_MODEL.append(frame)
    AVG += frame
AVG /= N



def conv_fg_mask():
    p=2
    global FG_MASK
    tmp = np.array(FG_MASK,copy=True)
    for i in range(width):
        for j in range(height):
            x = np.sum(FG_MASK[i-p:i+p,j-p:j+p])
            if x < 4*p*p*0.4:
                tmp[i][j] = 0
    FG_MASK = tmp

def update_fg_mask(B, I):
    global DISTMIN
    d_mean = np.average(DISTMIN)
    print("distance",d_mean)
    # updating decision threshold
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (R[i][j] > d_mean*R_scale):
                R[i][j] *= (1-R_inc_dec)
            else:
                R[i][j] *= (1+R_inc_dec)
    tmp = np.zeros((width, height), dtype='uint8')
    for k in range(N):
        dist = np.average((B[k] - I)**2,2)
        DISTMIN = dist#np.minimum(DISTMIN,dist)
        tmp += (dist < R).astype('uint8')

    global FG_MASK
    FG_MASK = (tmp < N_min).astype('uint8')

    #conv_fg_mask()

# continuous iteration
while (1):
    ret, frame = cap.read()
    cv2.imshow('ori', frame)
    update_fg_mask(BG_MODEL, frame)

    AVG *= N
    AVG -= BG_MODEL.pop(0)
    frame[:,:,0]*= 1 - FG_MASK
    frame[:,:,1]*= 1 - FG_MASK
    frame[:,:,2]*= 1 - FG_MASK
    AVG += frame
    AVG /= N
    BG_MODEL.append(frame)


    x =np.array(FG_MASK, float)
    cv2.imshow('frame', x)
    cv2.imshow('frame2', AVG/255)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
