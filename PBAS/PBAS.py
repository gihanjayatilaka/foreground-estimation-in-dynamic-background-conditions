import numpy as np
import cv2,sys




def conv_fg_mask(FG_MASK):
    p=2
    width, height = FG_MASK.shape[0], FG_MASK.shape[1]
    tmp = np.array(FG_MASK,copy=True)
    for i in range(width):
        for j in range(height):
            x = np.sum(FG_MASK[i-p:i+p,j-p:j+p])
            if x < 4*p*p*0.4:
                tmp[i][j] = 0
    return tmp

def update_fg_mask(B, I, R, DISTMIN):
    
    R_scale =5
    R_inc_dec = 0.01
    width, height = I.shape[0], I.shape[1]
    N = 30
    N_min = 2
    d_mean = np.average(DISTMIN)
    
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

    FG_MASK = (tmp < N_min).astype('uint8')

    FG_MASK = conv_fg_mask(FG_MASK)

    return FG_MASK

def pbas(filename):
    cap = cv2.VideoCapture(filename)
    _, frame = cap.read()
    N = 30
    N_min = 2
    width, height = frame.shape[0], frame.shape[1]
    

    BG_MODEL = []
    R = np.zeros((width, height))
    R+=100
    T = np.zeros((width,height)) # not added yet
    DISTMIN = np.zeros((width, height))


    for i in range(100):
        _, frame = cap.read()
        # first few frames
    for i in range(N):
        _, frame = cap.read()
        BG_MODEL.append(frame)
    

    # continuous iteration
    while (1):
        ret, frame = cap.read()
        cv2.imshow('ori', frame)
        FG_MASK = update_fg_mask(BG_MODEL, frame, R, DISTMIN)


        BG_MODEL.pop(0)
        frame[:,:,0] = frame[:,:,0]*(1 - FG_MASK)
        frame[:,:,1] = frame[:,:,1]*(1 - FG_MASK)
        frame[:,:,2] = frame[:,:,2]*(1 - FG_MASK)
        
        BG_MODEL.append(frame)


        x =np.array(FG_MASK, float)
        cv2.imshow('frame', x)
        

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pbas(sys.argv[1])
