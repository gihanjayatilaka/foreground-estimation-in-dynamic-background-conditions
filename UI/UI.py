import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import cv2
import sys
import argparse
import numpy as np
import signal
import textwrap
import warnings
#from scipy import ndimage
from my_filter import GB_filter, size_filter
from RPCA import inexact_augmented_lagrange_multiplier
from util_func import cvtcol, read_vid
from OF import OF

def signal_handler(signal, frame):
    cv2.destroyAllWindows()
    cap.release()
    if args.out_name:
        out_vid.release()
    sys.exit(0)

col_choice = ['rgb', 'r', 'g', 'b', 'gray']
#algo_choice = ['GMM', 'AGMM', 'GMG', 'RPCA', 'OF', 'AFCGMM']
algo_choice = ['GMM', 'AGMM', 'KNN', 'RPCA', 'OF', 'AFCGMM']
filter_choice = ['GB']
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))

parser = argparse.ArgumentParser(description='Bla Bla Bla.\n'
                                             'Note that some of the flags belong to certain algorithms and if incorrect falgs are used they will be ignored\n')
parser.add_argument('-i', '--input', dest='in_name',
                    help='Name of input video file. If unspecified input is read from camera')
parser.add_argument('-o', '--output', dest='out_name',
                    help='Name of output file.')
parser.add_argument('--colour', dest='colour', choices=col_choice,
                    help='Colour space to process video.')
parser.add_argument('--algorithm', dest='algo', choices = algo_choice,
                    help='''
                    Algorithm for background subtraction.\n
                    GMM (gaussian mixture model) \t - GMM gives deterministic values. \n
                    AGMM (adaptive gaussian mixture model) \t - AGMM gives probabilistic values between 0 and 255. Use --history flag to specify number of frames considered. Use --filter flag for morphological image processing.
                    ''')
parser.add_argument("--filter", nargs=3, action='append', metavar=('RADIUS','INTENSITY', 'SIZE'), type = float,
                    help = 'Apply gaussian blur and then low pass filter for intensity and size of components. Specify the blur radius, intensity limit and size limit.')
parser.add_argument('--show', action = 'store_true', dest='play_in', default=False,
                    help='Set flag to visualise input.')
parser.add_argument('--play', action = 'store_true', dest='play_vid', default=False,
                    help='Set flag to visualise output.')
parser.add_argument('-v', '--verbose', action = 'store_true', dest='verbose', default=False,
                    help='Set flag to view additional information.')
parser.add_argument('--start', dest='s_frame', type = int, default=0,
                    help='Starting frame number')
parser.add_argument('--end', dest='e_frame', type = int,
                    help='Ending frame number')
parser.add_argument('--lambda', dest='lmbda', type = float, default=.01,
                    help='Value of lambda for RPCA')
parser.add_argument('--iter', dest='iter', type = int, default=100,
                    help='Number of iterations for RPCA')
parser.add_argument('--clusters', dest='clusters', type = int, default=5,
                    help='Number of clusters for AFCGMM')
parser.add_argument('--history', dest='history', type = int, default=500, help='Number of previous frames to be used for AGMM')
#parser.add_argument('--bg', dest='background_ratio', type = float, default=.7,help='Background ratio for AGMM')
#parser.add_argument('--lr', dest='learning_rate', type = float, default=0, help='Learning rate for AGMM')

#EDIT THIS
#argument = '-i sample.mp4 -o out_temp.mp4 --algo AGMM --play --show --filter .5 200 3 --filter .5 150 3'.split()
#argument = '-i sample3.mp4 -o out_temp.mp4 --algo RPCA --play --show --filter .5 200 3 --filter .5 150 3 --frame 100 --pp gray'.split()
argument = '-i ori.avi --algo AGMM --play --show --filter 0 200 3 --filter 0.5 200 5'.split()

args = parser.parse_args()
#args = parser.parse_args(argument)
#parser.print_help()
#print(args)
#-----------------------------------------------------------------------------------
#Default
RT = True
rows, cols, o = None, None, None
n_frame = args.s_frame

if args.e_frame == None:
    args.e_frame = float('inf')

#Check input
cap = read_vid(args.in_name)

#VideoCapture valid???
if (cap.isOpened()== False):
    #raise Error("Error opening video stream or file")
    sys.exit(0)
    #os._exit(0)
else:
    ret, frame = cap.read()
    rows, cols, _ = frame.shape

for i in range(args.s_frame):
    ret, frame = cap.read()

if args.out_name:
    out_vid = cv2.VideoWriter(args.out_name, -1, 20.0, (cols, rows), False)
else:
    out_vid = None

# Algo
if args.algo == None:
    print ("Please Enter an algorithm")
    sys.exit(0)
elif args.algo == 'GMM':
    fgbg= cv2.bgsegm.createBackgroundSubtractorMOG()
elif args.algo == 'AGMM':
    fgbg= cv2.createBackgroundSubtractorMOG2(history = args.history)
elif args.algo == 'KNN':
    fgbg= cv2.createBackgroundSubtractorKNN()
elif args.algo == 'RPCA':
    RT = False
elif args.algo == 'AFCGMM':
    cv2.destroyAllWindows()
    cap.release()
    if args.out_name:
        out_vid.release()
    os.system("python afcgmm.py " + str(args.in_name) + " " + str(args.clusters) + " " + str(args.verbose) + " " + str(args.play_in) + " " + str(args.play_vid) + " " + str(args.out_name))
elif args.algo == 'OF':
    out = np.zeros_like(frame)
    out[...,1] = 255
    prev = frame

signal.signal(signal.SIGINT, signal_handler)

if args.algo == 'AFCGMM':
    pass
elif RT:
    while cap.isOpened():
        #frame_limit
        if n_frame < args.e_frame:
            n_frame+= 1
        else:
            break

        ret, frame = cap.read()

        if ret == True:

            # colour space
            if args.algo != 'OF':
                frame = cvtcol(frame, args.colour)

            # play input
            if args.play_in:
                cv2.imshow('Input', frame)

            # Algo
            if args.algo == 'OF':
                out = OF(prev, frame, out)
                prev = frame
            else:
                FG = fgbg.apply(frame)
                out = FG

                # Filter
                if args.filter and n_frame>args.s_frame+5:
                    for filter in args.filter:
                        GB_FG = GB_filter(out, filter[0], filter[1], mask=1, mask_in=FG)
                        out = size_filter(GB_FG, filter[2], mask=1, mask_in=FG)

#            out = cv2.morphologyEx(FG, cv2.MORPH_CLOSED, kernel)

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

else:
    mat = []

    while cap.isOpened():
        #frame_limit
        if n_frame < args.n_frame:
            n_frame+= 1
        else:
            break

        ret, frame = cap.read()

        if ret == True:
            # colour space
            frame = cvtcol(frame, args.colour)

            # play input
            if args.play_in:
                cv2.imshow('Input', frame)

            mat.append(np.ndarray.flatten(frame))

            if args.play_in:
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

    matrix = np.asarray(mat).T

    #print(matrix.shape)

    A, E = inexact_augmented_lagrange_multiplier(matrix, lmbda=args.lmbda, maxiter = args.iter, verbose = args.verbose)

    _, n_frame = E.shape
    FG_mat = np.reshape(E.T.astype(np.uint8), (n_frame, rows, cols))
    #print(FG.shape)

    for i in range(n_frame):
        FG = FG_mat[i, :, :]
        out = FG

        # Filter
        if args.filter:
            for filter in args.filter:
                GB_FG = GB_filter(out, filter[0], filter[1], mask=1, mask_in=FG)
                out = size_filter(GB_FG, filter[2], mask=1, mask_in=FG)

        if args.play_vid:
            cv2.imshow('output', out)

        if args.play_vid:
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        if args.out_name:
            out_vid.write(out)




cv2.destroyAllWindows()
cap.release()

if args.out_name:
    out_vid.release()




'''
class ValidateCredits(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # print '{n} {v} {o}'.format(n=args, v=values, o=option_string)
        valid_subjects = ('foo', 'bar')
        subject, credits = values
        if subject not in valid_subjects:
            raise ValueError('invalid subject {s!r}'.format(s=subject))
        credits = float(credits)
        Credits = collections.namedtuple('Credits', 'subject required')
        setattr(args, self.dest, Credits(subject, credits))'''
