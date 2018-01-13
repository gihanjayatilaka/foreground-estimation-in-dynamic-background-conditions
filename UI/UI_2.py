import cv2
import argparse
import numpy as np
import os
from scipy import ndimage
from my_filter import GB_filter, size_filter


col_choice = ['rgb', 'r', 'g', 'b', 'gray']
algo_choice = ['GMM', 'AGMM', 'GMG', 'RPCA']
filter_choice = ['GB']

parser = argparse.ArgumentParser(description='Bla Bla Bla')
parser.add_argument('-i', '--input', dest='in_name',
                    help='Name of input video file. If unspecified input is read from camera')
parser.add_argument('-o', '--output', dest='out_name',
                    help='Name of output file.')
parser.add_argument('--pp', dest='pp', choices=col_choice,
                    help='Colour space to process video.')
parser.add_argument('--algorithm', dest='algo', choices = algo_choice,
                    help='Algorithm for background subtraction')
parser.add_argument("--filter", nargs=3, action='append', metavar=('RADIUS','INTENSITY', 'SIZE'), type = float,
                    help = 'Apply gaussian blur and then low pass filter for intensity and size of components. Specify the blur radius, intensity limit and size limit.')
parser.add_argument('--show', action = 'store_true', dest='play_in', default=False,
                    help='Set flag to visualise input.')
parser.add_argument('--play', action = 'store_true', dest='play_vid', default=False,
                    help='Set flag to visualise output.')

#EDIT THIS
argument = '-i sample.mp4 -o out_temp.mp4 --algo AGMM --play --show --filter .5 200 3 --filter .5 150 3'.split()


args = parser.parse_args(argument)
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
    fourcc = -1
    out_vid = cv2.VideoWriter(name, fourcc, 20.0, (cols, rows), False)


# Algo
if args.algo == 'GMM':
    fgbg = cv2.BackgroundSubtractorMOG()
elif args.algo == 'AGMM':
    fgbg= cv2.createBackgroundSubtractorMOG2()
elif args.algo == 'GMG':
    fgbg= cv2.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


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
        out = FG
        cv2.imshow('out', out)

        if args.algo == 'GMG':
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

        # Filter
        for filter in args.filter:
            GB_FG = GB_filter(out, filter[0], filter[1], mask=1, mask_in=FG)
            out = size_filter(GB_FG, filter[2], mask=1, mask_in=FG)

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
