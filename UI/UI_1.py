import cv2
import argparse
import numpy as np
import os
from scipy import ndimage

def GB_filter(FG, blur_radius, threshold, mask = 0, mask_in = None):
    GB_FG = ndimage.gaussian_filter(FG, blur_radius)
    if mask:
        return np.where(GB_FG > threshold, mask_in, 0).astype(np.uint8)
    else:
        return np.where(GB_FG > threshold, 255, 0).astype(np.uint8)

def size_filter(FG, min_size, mask = 0, mask_in = None):
    ''' m, n = FG.shape
    labeled, num_features = ndimage.measurements.label(FG, structure=np.ones((3,3)))
    unique, lab, count = np.unique(labeled, return_inverse=True, return_counts=True)
    lab = lab.reshape(m, n)
    bg_mask = np.where(count>min_size, 1, 0)
    bg_mask[0] = 0
    F_FG = bg_mask[lab].reshape(m, n) '''

    labeled, num_features = ndimage.measurements.label(FG, structure=np.ones((3,3)))
    unique, count = np.unique(labeled, return_counts=True)
    bg_mask = np.where(count>min_size, 1, 0)
    bg_mask[0] = 0
    F_FG = bg_mask[labeled]

    if mask:
        return F_FG.astype(np.uint8)*mask_in
    else:
        return F_FG.astype(np.uint8)*255






col_choice = ['rgb', 'r', 'g', 'b', 'gray']
algo_choice = ['GMM', 'AGMM', 'GMG', 'RPCA']
filter_choice = ['GB']

parser = argparse.ArgumentParser(description='Bla Bla Bla')
parser.add_argument('-i', '--input', dest='in_name',
                    help='Name of input video file. If unspecified input is read from camera')
parser.add_argument('-o', '--output', dest='out_name',
                    help='Name of output file.')
#parser.add_argument('--algorithm', dest='algo_list', nargs = '+',
#                    help='List of algorithms in the order of implementation.')
parser.add_argument('--pp', dest='pp', choices=col_choice,
                    help='Colour space to process video.')
parser.add_argument('--show', action = 'store_true', dest='play_in', default=False,
                    help='Set flag to visualise input.')
parser.add_argument('--play', action = 'store_true', dest='play_vid', default=False,
                    help='Set flag to visualise output.')
parser.add_argument('--algorithm', dest='algo', choice = algo_choice,
                    help='Algorithm for background subtraction')

argument = '-i sample.mp4 -o out_temp.mp4 --algo AGMM --play --show --pp g'.split()
args = parser.parse_args(argument)
#print (args.accumulate(args.integers))
parser.print_help()
print(args)





#Check input
if args.in_name:
    cap = cv2.VideoCapture(args.in_name)
else:
    cap = cv2.VideoCapture(0)

#VideoCapture valid???
if (cap.isOpened()== False):
  print("Error opening video stream or file")
else:
    ret, frame = cap.read()
    m, n, o = frame.shape

if args.out_name:
    rows, cols = m, n
    name = args.out_name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    except:
        fourcc = cv2.CV_FOURCC(*'XVID')
    out_vid = cv2.VideoWriter(name, fourcc, 20.0, (cols, rows), False)


# Algo
if args.algo == 'AGMM':
    fgbg= cv2.createBackgroundSubtractorMOG()
elif args.algo == 'GMM':
    fgbg= cv2.createBackgroundSubtractorMOG2()
elif args.algo == 'GMG':
    fgbg= cv2.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

'''
#GMM
GMM = cv2.createBackgroundSubtractorMOG()
#AGMM
AGMM = cv2.createBackgroundSubtractorMOG2()
#GMG
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
GMG = cv2.createBackgroundSubtractorGMG()
'''



while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:

        # colour space
        if args.pp == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif args.pp == 'r':
            frame = frame[:, :, 0]
        elif args.pp == 'g':
            frame = frame[:, :, 1]
        elif args.pp == 'b':
            frame = frame[:, :, 2]

        # play input
        if args.play_in:
            cv2.imshow('Input', frame)

        # Algo
        FG = fgbg.apply(frame)

        if args.algo == 'GMG':
            FG = cv2.morphologyEx(FG, cv2.MORPH_OPEN, kernel)

        GB_FG1 = GB_filter(FG, .5, 200, mask=1, mask_in=FG)
        S_F1 = size_filter(GB_FG1, 3, mask=1, mask_in=FG)
#        cv2.imshow('S_F1', S_F1)

        GB_FG1 = GB_filter(FG, .5, 150, mask=1, mask_in=FG)
        S_F2 = size_filter(GB_FG1, 3, mask=1, mask_in=FG)
#        cv2.imshow('S_F2', S_F2)

#        cv2.imshow('diff', S_F2-S_F1)

        out = S_F1

        if args.play_vid:
            cv2.imshow('output', out)

        if args.play_in or args.play_vid:
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        if args.out_name:
            out_vid.write(out)

    else:
        break

cv2.destroyAllWindows()
cap.release()

if args.out_name:
    out_vid.release()




'''
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    except:
        fourcc = cv2.CV_FOURCC(*'XVID')

    rows, cols = m, n

    try:
        n_frame, rows, cols, color = output.shape
        out_vid = cv2.VideoWriter(name, fourcc, 20.0, (cols, rows), True)

        for i in range(n_frame):
            img = output[i, :, :, :]
            out_vid.write(img)
            cv2.imshow('boat', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.destroyAllWindows()

        out_vid.release()

    except:
        n_frame, rows, cols = output.shape
        out_vid = cv2.VideoWriter(name, fourcc, 20.0, (cols, rows), False)

        for i in range(n_frame):
            img = output[i, :, :]
            out_vid.write(img)
            cv2.imshow('boat', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.destroyAllWindows()

        out_vid.release()
    '''


'''
        GB_FG1 = GB_filter(FG, .5, 200, mask=1, mask_in=FG)
        cv2.imshow('GB_FG1', GB_FG1)

        m = np.amax(GB_FG1)
        adj = np.where(GB_FG1>0, GB_FG1, m)
        print(np.amin(adj), np.amax(adj))

        S_F1 = size_filter(GB_FG1, 3, mask=1, mask_in=FG)
        cv2.imshow('S_F1', S_F1)
        '''

'''
        #rgb2gray
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Frame',f)

        mat.append(np.ndarray.flatten(f))
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        '''

