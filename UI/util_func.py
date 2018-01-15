import cv2

def cvtcol(frame, pp):
    if pp == 'gray':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif pp == 'r':
        frame = frame[:, :, 0]
    elif pp == 'g':
        frame = frame[:, :, 1]
    elif pp == 'b':
        frame = frame[:, :, 2]
    return frame

def read_vid(name):
    if name:
        return cv2.VideoCapture(name)
    else:
        return cv2.VideoCapture(0)

