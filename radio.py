from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError
from scipy import signal
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import filters as im_filters
import numpy as np
import encoding as enc

import matplotlib.pyplot as plt

TRANS_SIZE = 7500

# Compression types
NO_COMPRESSION = 0
DECIMATE = 1
BLACK_WHITE = 2

send_queue = Queue()

class Transmitter(object):

    def __init__(self):
        pass

    def transmit(self, image, imtype=None):
        """
        Compresses and transmits image in 75 seconds
        """
        img_comp, comp_type = self.compress_image(image, imtype)
        bits = self.to_packet_bits(img_comp, comp_type)

        Process(target=self.send_bits, args=(bits, )).start()


    def compress_image(self, image, imtype):
        """
        Compresses the image so it will fit in a packet. Returns an integer
        for the type of compression it uses.
        """
        if imtype is None:
            if image.shape[2] * image.shape[0]*image.shape[1] <= TRANS_SIZE:
                imtype = NO_COMPRESSION
            else:
                imtype = DECIMATE
        
        if imtype == NO_COMPRESSION:
            return image, NO_COMPRESSION
        elif imtype == BLACK_WHITE:
            gs = np.average(image, axis=2)
            black_loc = np.unravel_index(gs.argmin(), gs.shape)
            white_loc = np.unravel_index(gs.argmax(), gs.shape)
            black, white = image[black_loc[0]][black_loc[1]], image[white_loc[0]][white_loc[1]]

            bw = (gs > np.max(gs)/2).astype(int)
            bw_1D = np.reshape(bw, bw.shape[0]*bw.shape[1])
            rle_bw = bw_rle_encode(bw_1D)

            return {'rle_image': rle_bw, 'black': black, 'white': white, 'shape': image.shape}, BLACK_WHITE

        elif imtype == DECIMATE:
            bytes_original = 3 * image.shape[0] * image.shape[1]
            down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))
            if down_sampling_factor == 1:
                return image, NO_COMPRESSION

            h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
            image_lpf = convolve_image(image, h_lpf)
            decimated_xy = image_lpf[::down_sampling_factor, ::down_sampling_factor]

            return {'decimated_image': decimated_xy, 'shape': image.shape}, DECIMATE

        return None

    def to_packet_bits(self, img_comp, comp_type):
        """
        Turns the compressed image into a bitstream packet
        """
        if comp_type == DECIMATE:
            img = img_comp['decimated_image']
            img_shape = img.shape
            img_1D = np.reshape(img, img_shape[0]*img_shape[1]*img_shape[2])
            img_bytes = img_1D.tobytes()

            img_comp['decimated_image'] = img_shape, enc.encode(img_bytes)
            print('Sending', len(img_comp['decimated_image'][1]), 'bytes.')

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
        elif comp_type == DECIMATE:
            img_shape, img_bytes = image_comp['decimated_image']
            img_1D = np.fromstring(enc.decode(img_bytes), dtype=np.uint8)
            decimated = np.reshape(img_1D, img_shape)
            orig_shape = image_comp['shape']

            bytes_original = 3 * orig_shape[0] * orig_shape[1]
            down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))

            upsampled = imresize(decimated, orig_shape, interp='nearest').astype(np.uint8)
            h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
            upsample_xy = convolve_image(upsampled, h_lpf)

            return upsample_xy
        elif comp_type == BLACK_WHITE:
            rle_bw, black, white, orig_shape = image_comp['rle_image'], image_comp['black'], image_comp['white'], image_comp['shape']

            bw_1D = bw_rle_decode(rle_bw)
            bw = np.reshape(bw_1D, (orig_shape[0], orig_shape[1]))
            bw_inv = (bw < 1).astype(np.uint8)
            red_w, green_w, blue_w = bw*255, bw*255, bw*255
            red_b, green_b, blue_b = bw_inv*0, bw_inv*0, bw_inv*0
            red, green, blue = red_w + red_b, green_w + green_b, blue_w + blue_b

            recon = np.zeros(orig_shape).astype(np.uint8)
            recon[:,:,0], recon[:,:,1], recon[:,:,2] = red, green, blue

            return recon


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
    image = np.copy(image)
    image = image / np.max(image)
    image *= 255
    return image.astype(np.uint8)

def bw_rle_encode(bw):
    encoded = []
    curr_val, count = 0, 0
    for pix in bw:
        if pix != curr_val:
            encoded.append(count)
            curr_val, count = pix, 0
        count += 1
    encoded.append(count)

    return np.array(encoded)

def bw_rle_decode(bw_rle):
    decoded = []
    curr_val = 0
    for run in bw_rle:
        decoded.extend([curr_val]*run)
        curr_val = 0 if curr_val else 1  # flip curr_val
    return np.array(decoded).astype(np.uint8)

## Shitty Decimation
def simple_decimation(image):
    bytes_original = float(image.shape[0]) * float(image.shape[1])
    down_sampling_factor = np.ceil(bytes_original / TRANS_SIZE)
    block_size = int(np.ceil(np.sqrt(down_sampling_factor)))
   
    downsample0 = signal.decimate(np.asarray(image, dtype=np.float), block_size, axis = 0) 
    downsample1 = np.asarray( signal.decimate(downsample0, block_size, axis = 1), dtype=np.uint8)           
    return (downsample1, image.shape)

def simple_upsample(image, shape):
    bytes_original = float(shape[0]) * float(shape[1])
    down_sampling_factor = np.ceil(bytes_original / TRANS_SIZE)
    block_size = int(np.ceil(np.sqrt(down_sampling_factor)))
    print(image.shape) 
    upsample0 = signal.resample(image, shape[0], axis = 0) 
    print(upsample0.shape) 
    upsample1 = signal.resample(upsample0, shape[1], axis = 1)             
    print(upsample1.shape) 
    return upsample1

# TRANSMISSION / RECIEVE BIT STREAM    
# bit_stream = np.unpackbits(downsample1)
# image = np.packbits(bit_stream) 

