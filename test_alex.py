import cv2
import scipy
from scipy.misc import imread
from scipy import fftpack
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import urllib2

import radio

img = imread('imgs/Marconi.tiff')
#img = img[:,:,0]
dwn_sample = radio.dct_downsample(img, 0)
imgreturn = radio.dct_upsample(dwn_sample, img.shape, 0)
plt.imshow(imgreturn)
plt.show()
