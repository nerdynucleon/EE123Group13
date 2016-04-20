from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError
from scipy import signal
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import filters as im_filters
import numpy as np

import matplotlib.pyplot as plt

TRANS_SIZE = 2500

# Compression types
NO_COMPRESSION = 0
DECIMATE = 1

send_queue = Queue()

class Transmitter(object):

    def __init__(self):
        pass

    def transmit(self, image):
        """
        Compresses and transmits image in 75 seconds
        """
        # image = np.average(image, axis=2).astype(int)

        img_comp, comp_type = self.compress_image(image)
        bits = self.to_packet_bits(img_comp, comp_type)

        Process(target=self.send_bits, args=(bits, )).start()


    def compress_image(self, image):
        """
        Compresses the image so it will fit in a packet. Returns an integer
        for the type of compression it uses.
        """

        bytes_original = 3 * image.shape[0] * image.shape[1]
        down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))
        if down_sampling_factor == 1:
            return image, NO_COMPRESSION

        h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
        image_lpf = convolve_image(image, h_lpf)
        decimated_xy = image_lpf[::down_sampling_factor, ::down_sampling_factor]

        return {'decimated_image': decimated_xy, 'shape': image.shape}, DECIMATE


    def to_packet_bits(self, img_comp, comp_type):
        """
        Turns the compressed image into a bitstream packet
        """

        return img_comp, comp_type


    def send_bits(self, bits):
        """
        Sends bits through Baofeng.
        """

        send_queue.put(bits)


class Receiver(object):

    def __init__(self):
        pass

    def receive(self):
        """
        Recieves images and returns them.
        """

        bits = self.read_bits()
        pkt = self.decode_packet(bits)
        image = self.packet_to_img(pkt)
        return image


    def decode_packet(self, bits):
        """
        Decodes the packet bits into a packet.
        """
        image_comp, comp_type = bits
        if comp_type == NO_COMPRESSION:
            return image_comp

        decimated, orig_shape = image_comp['decimated_image'], image_comp['shape']

        upsample_xy = imresize(decimated, orig_shape).astype(np.uint8)

        return upsample_xy


    def packet_to_img(self, pkt):
        """
        Turns the packet into an image.
        """


        return pkt


    def read_bits(self):
        """
        Read bits from SDR
        """

        try:
            return send_queue.get(timeout=75)
        except QueueEmptyError:
            return None

def wc_to_sigma(wc):
    return np.sqrt(2*np.log(2))/wc

def gaussian_lpf(wc, width=None):
    sigma = wc_to_sigma(wc)
    if width is None:
        width = 6*sigma

    x = np.r_[:width].astype(float)
    y = x[:,np.newaxis]
    x0 = y0 = width // 2
    x, y = x-x0, y-y0

    return np.exp(-(x**2 + y**2)/(2*sigma**2)) / (2 * np.pi * sigma**2)

def convolve_image(image, filt):
    image_lpf = np.zeros(image.shape)
    for i in range(3):
        image_lpf[:,:,i] = im_filters.convolve(image[:,:,i], filt) # signal.convolve2d(image[:,:,i], filt, mode='same')

    return image_lpf.astype(np.uint8)

def to_uint8(image):
    image += np.min(image)
    image *= 255/np.max(image)
    return image.astype(np.uint8)
        
