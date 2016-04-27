from __future__ import division
from __future__ import print_function
import scipy
from scipy.misc import imread, imsave
from radio import Transmitter, Receiver
import matplotlib.pyplot as plt
import sys
from scipy import signal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def PSNR_calculator(original, compressed):
    MSE = np.sum(np.square(np.asarray(original, dtype=np.float) - np.asarray(compressed, dtype=np.float))) 
    MSE = MSE / (float(original.shape[0]) * float(original.shape[1]) * float(original.shape[2]))
    if MSE == 0:
        return float('inf')
    return 20 * np.log10( float(np.iinfo(original.dtype).max) + 1) - 10 * np.log10(MSE)

def main(img_files = ['imgs/im1.bmp', 'imgs/im2.bmp'], display = True, verbose = True):    
    if(len(sys.argv) > 1):
        img_files = sys.argv[1:]
    
    imgs = []
    for img_file in img_files:
        try:
            imgs.append(imread(img_file))
        except IOError:
            print('Unable to open file:', img_file, file=sys.stderr)
            return

    tx, rx = Transmitter(), Receiver()
    for i in range(len(imgs)):
        img = imgs[i]
        tx.transmit(img)
        img_rcv = rx.receive()
        try:
            psnr = PSNR_calculator(img, img_rcv)
            if verbose:
                print('PSNR for {}: {} db'.format(img_files[i], psnr))
            
            if display:
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
