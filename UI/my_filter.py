import numpy as np
from scipy import ndimage
import argparse

def GB_filter(FG, blur_radius, threshold, mask = 0, mask_in = None):
    GB_FG = ndimage.gaussian_filter(FG, blur_radius)
    if mask:
        return np.where(GB_FG > threshold, mask_in, 0).astype(np.uint8)
    else:
        return np.where(GB_FG > threshold, 255, 0).astype(np.uint8)

def size_filter(FG, min_size, mask = 0, mask_in = None):
    labeled, num_features = ndimage.measurements.label(FG, structure=np.ones((3,3)))
    unique, count = np.unique(labeled, return_counts=True)
    bg_mask = np.where(count>min_size, 1, 0)
    bg_mask[0] = 0
    F_FG = bg_mask[labeled]

    if mask:
        return F_FG.astype(np.uint8)*mask_in
    else:
        return F_FG.astype(np.uint8)*255

'''
class ValidateFilter(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('name', namespace)
        print('val', values)
        message = ''
        if len(values) == 3:
            try:
                values[0] = float(values[0])
                if values[0] < 0:
                    message = 'first value (blur radius) of argument "{}" must be positive'.format(self.dest)
            except ValueError:
                message = 'first value (blur radius) of argument "{}" must be float'.format(self.dest)

            try:
                values[1] = int(values[1])
                if values[1] < 0 or values[1] > 255:
                    message = 'second value (intensity constraint) of argument "{}" must be between 0 and 255'.format(self.dest)
            except ValueError:
                message = 'second value (intensity constraint) of argument "{}" must be integer'.format(self.dest)

            try:
                values[2] = int(values[2])
                if values[2] < 0:
                    message = 'third value (size constraint) of argument "{}" must be non negative'.format(self.dest)
            except ValueError:
                message = 'third value (size constraint) of argument "{}" must be integer'.format(self.dest)
        else:
            message = 'argument "{}" requires 3 values'.format(self.dest)
        if message:
            raise argparse.ArgumentError(self, message)
        setattr(namespace, self.dest, values)
'''

'''
class ValidateFilter(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        valid_subjects = ('foo', 'bar')
        subject, credits = values
        if subject not in valid_subjects:
            raise ValueError('invalid subject {s!r}'.format(s=subject))
        credits = float(credits)
        Credits = collections.namedtuple('Credits', 'subject required')
        setattr(args, self.dest, Credits(subject, credits))

class StringInteger(argparse.Action):
        """Action to assign a string and optional integer"""
        def __call__(self, parser, namespace, values, option_string=None):
            message = ''
            if len(values) not in [1, 2]:
                message = 'argument "{}" requires 1 or 2 arguments'.format(
                    self.dest)
            if len(values) == 2:
                try:
                    values[1] = int(values[1])
                except ValueError:
                    message = ('second argument to "{}" requires '
                               'an integer'.format(self.dest))
            else:
                values.append(int_default)
            if message:
                raise argparse.ArgumentError(self, message)
            setattr(namespace, self.dest, values)
'''