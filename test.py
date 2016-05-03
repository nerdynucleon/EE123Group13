import cv2
import scipy
from scipy.misc import imread, imsave
from scipy import fftpack
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import urllib2

import io

CARTOON_THRESHOLD = 50

#imgs = ['imgs/Isee.tiff', 'imgs/Marconi.tiff', 'imgs/calBlue.tiff', 'imgs/pauly.tiff']
imgs = ['imgs/calBlue.tiff']
# imgs.append(imread('imgs/Isee.tiff'))
# imgs.append(imread('imgs/Marconi.tiff'))
# imgs.append(imread('imgs/calBlue.tiff'))
# imgs.append(imread('imgs/pauly.tiff'))
# imgs.append(Image.open('imgs/Marconi.tiff'))


# image = np.array(imgs[0], dtype=np.float)
# # for image in imgs:
#     # image_smoothed = cv2.bilateralFilter(image, 4, 20, 20)
#     #
#     # #the bigger the difference the less cartoon like it is
#     # cartoon_index = np.mean(image - image_smoothed)
#     # print cartoon_index
#     # if cartoon_index > CARTOON_THRESHOLD:
#     #     print("real")
#     # else:
#     #     print("cartoon")
#
# dct_im = fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
# num_coeffs_dropped = 60#image.shape[0]/2
# print num_coeffs_dropped
# dct_compress = dct_im.copy()
# dct_compress[num_coeffs_dropped:,:] = 0
# dct_compress[:,num_coeffs_dropped:] = 0
# image_reconstruct = fftpack.idct(fftpack.idct(dct_compress.T, norm='ortho').T, norm='ortho')
# image_reconstruct = image_reconstruct.clip(0,255)
# image_reconstruct = image_reconstruct.astype('uint8')
# # image_fin = Image.fromarray(image_reconstruct)
#
# # scipy.misc.imsave('outfile.jpg', image_reconstruct)
# plt.imshow(image_reconstruct)
# plt.show()

# image_url='http://i.imgur.com/8vuLtqi.png'

for img_name in imgs:
    file_descriptor = open(img_name)
    image_file = io.BytesIO(file_descriptor.read())
    image = imread(img_name)#Image.open(image_file)
    # img_color = image.resize(size, 1)
    # img_r = img_color[:,:,0]
    img_hsv = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)#image.convert('L')
    img = np.array(img_hsv, dtype=np.float)

    print img.shape


    dct_im = fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')
    dct_copy = dct_im.copy()
    print("dct shape", dct_copy.shape)
    print(dct_copy)
    dct_copy[64:,:] = 0
    dct_copy[:,64:] = 0
    raw = fftpack.idct(fftpack.idct(dct_copy.T, norm='ortho').T, norm='ortho')
    img_f = raw.clip(0, 255)
    img_f = img_f.astype('uint8')
    img_f = cv2.cvtColor(img_f, cv2.cv.CV_HSV2BGR)
    img_f = Image.fromarray(img_f)
    scipy.misc.imsave('outfile.jpg', img_f)

# plt.imshow(img_f)
# plt.show()
