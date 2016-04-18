#!/usr/bin/python

from __future__ import division
import sys
from scipy import signal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

# 2500 pixels to work with
transmission_size = 2500.0

def PSNR_calculator(original, compressed):
    MSE = np.sum(np.square(np.asarray(original, dtype=np.float) - np.asarray(compressed, dtype=np.float))) 
    MSE = MSE / (float(original.shape[0]) * float(original.shape[1]) * float(original.shape[2]))
    return 20.0 * np.log10( float(np.iinfo(original.dtype).max) + 1.0 ) - 10.0 * np.log10( MSE )

def simple_encode_downsample(image):
    global transmission_size
    bytes_original = float(image.shape[0]) * float(image.shape[1])
    down_sampling_factor = np.ceil(bytes_original / transmission_size)
    block_size = int(np.ceil(np.sqrt(down_sampling_factor)))
   
    downsample0 = signal.decimate(np.asarray(image, dtype=np.float), block_size, axis = 0) 
    downsample1 = np.asarray( signal.decimate(downsample0, block_size, axis = 1), dtype=np.uint8)           
    bit_stream = np.unpackbits(downsample1)
    return (bit_stream, image.shape)

def simple_decode_downsample(bit_stream, shape):
    global transmission_size
    bytes_original = float(shape[0]) * float(shape[1])
    down_sampling_factor = np.ceil(bytes_original / transmission_size)
    block_size = int(np.ceil(np.sqrt(down_sampling_factor)))
   
    image = np.packbits(bit_stream) 
    
    upsample0 = signal.resample(image, block_size, axis = 0) 
    upsample1 = signal.resample(upsample0, block_size, axis = 1)             
    return upsample1

def main():
    arguments = sys.argv
    if(len(arguments) < 2):
        print 'Failed to Supply Input Image'
        return -1
    else:
        im = misc.imread(arguments[1])
        bit_stream, shape = simple_encode_downsample(im)
        print bit_stream
        print shape
        image = simple_decode_downsample(bit_stream, shape)
        plt.imshow(image)
main()
