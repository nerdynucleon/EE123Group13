#!/usr/bin/python

from __future__ import division
import sys
from scipy import misc
import numpy as np

def PSNR_calculator(original, compressed):
    MSE = np.sum(np.square(np.asarray(original, dtype=np.float) - np.asarray(compressed, dtype=np.float))) 
    MSE = MSE / (float(original.shape[0]) * float(original.shape[1]) * float(original.shape[2]))
    return 20.0 * np.log10( float(np.iinfo(original.dtype).max) + 1.0 ) - 10.0 * np.log10( MSE )

def simple_downsample(image):
    return 0

def main():
    arguments = sys.argv
    if(len(arguments) < 2):
        print 'Failed to Supply Input Image'
        return -1
    else:
        im1 = misc.imread(arguments[1])
        im2 = misc.imread(arguments[2])
        print PSNR_calculator(im1,im2)

main()
