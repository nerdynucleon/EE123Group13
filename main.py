#!/usr/bin/python

import sys
from scipy import misc
import numpy as np

def PSNR_calculator(original, compressed):
    MSE = 0
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            for k in range(original.shape[2]):
                MSE = MSE + (original[i,j,k] - compressed[i,j,k])**2
    return 10*np.log(((3**original.shape[2])**2)*original.shape[0]*original.shape[1]/MSE)

def simple_downsample(image):
    return 0

def main():
    arguments = sys.argv
    if(len(arguments) < 2):
        print 'Failed to Supply Input Image'
        return -1
    else:
        im = misc.imread(arguments[1])
        print PSNR_calculator(im,im)

main()
