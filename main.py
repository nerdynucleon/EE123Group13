from __future__ import division
from __future__ import print_function
import scipy
import time
from scipy.misc import imread, imsave
import radio
import matplotlib.pyplot as plt
import sys
from scipy import signal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import argparse

def PSNR_calculator(original, compressed):
    MSE = np.sum(np.square(np.asarray(original, dtype=np.float) - np.asarray(compressed, dtype=np.float))) 
    MSE = MSE / (float(original.shape[0]) * float(original.shape[1]) * float(original.shape[2]))
    if MSE == 0:
        return float('inf')
    return 20 * np.log10( float(np.iinfo(original.dtype).max) + 1) - 10 * np.log10(MSE)

def main(img_files = ['imgs/im1.bmp', 'imgs/im2.bmp'], display = True, verbose = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='{rx, tx, test}')
    parser.add_argument('-t', '--im_type')
    parser.add_argument('fname')
    args = parser.parse_args()
    
    mode = args.mode if args.mode else 'test'
    im_type = args.im_type
    if im_type == 'NO_COMP':
        im_type = radio.NO_COMPRESSION
    elif im_type == 'DEC':
        im_type = radio.DECIMATE
    elif im_type == 'BW':
        im_type = radio.BLACK_WHITE
    else:
        im_type = None
    img_file = args.fname

    if mode == 'rx':
        rx = radio.Receiver()
        img = imread(img_file)
        img_rcv = rx.receive()
        try:
            imsave('rcv_image{}.tiff'.format(time.time()), img_rcv)

            psnr = PSNR_calculator(img, img_rcv)
            print('PSNR:', psnr, 'db')
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img)
            ax1.set_title('Sent image')
            ax2.imshow(img_rcv)
            ax2.set_title('Received image')

            plt.show()

        except:
            print('ERROR: Unable to display received image.', file=sys.stderr)
            raise
    elif mode == 'tx':
        img_files = sys.argv[1:]
        
        imgs = imread(img_file)

        tx = radio.Transmitter()
        tx.transmit(img, im_type)
    else:
        tx, rx = radio.Transmitter(lp_mode=True), radio.Receiver(lp_mode=True)

        img = imread(img_file)
        tx.transmit(img, im_type)
        img_rcv = rx.receive()
        try:
            psnr = PSNR_calculator(img, img_rcv)
            print('PSNR:', psnr, 'db')
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img)
            ax1.set_title('Sent image')
            ax2.imshow(img_rcv)
            ax2.set_title('Received image')

            plt.show()

        except:
            print('ERROR: Unable to display received image.', file=sys.stderr)
            raise

if __name__ == "__main__":
    main()
