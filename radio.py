from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError
from scipy import signal
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import filters as im_filters
# from transmit import send_bytes, receive_bytes
import numpy as np
import encoding as enc
import struct
import bitarray
import matplotlib.pyplot as plt

TRANS_SIZE = 7500

# Compression types
NO_COMPRESSION = 0
DECIMATE = 1
BLACK_WHITE = 2

send_queue = Queue()


# Struct Formats
COMP_TYPE = 'b'
COMP_TYPE_SIZE = struct.calcsize(COMP_TYPE)
NO_COMP_HEADER = '3i'
NO_COMP_HEADER_SIZE = struct.calcsize(NO_COMP_HEADER)
DECIMATE_HEADER = '6i'
DECIMATE_HEADER_SIZE = struct.calcsize(DECIMATE_HEADER)
BW_HEADER = '12ib'
BW_HEADER_SIZE = struct.calcsize(BW_HEADER)

class Transmitter(object):

    def __init__(self, lp_mode):
        self._lp_mode = lp_mode

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
            if image.shape[0]*image.shape[1]*image.shape[2] <= TRANS_SIZE:
                imtype = NO_COMPRESSION
            else:
                imtype = DECIMATE
        
        if imtype == NO_COMPRESSION:
            return image, NO_COMPRESSION
        elif imtype == BLACK_WHITE:
            dec = 1
            while True:
                image_dec, dec_shape = image, image.shape
                if dec != 1:
                    print('dec:', dec)
                    image_dec = image[::dec, ::dec]
                    dec_shape = image_dec.shape

                gs = np.average(image_dec, axis=2)
                black_loc = np.unravel_index(gs.argmin(), gs.shape)
                white_loc = np.unravel_index(gs.argmax(), gs.shape)
                black, white = image[black_loc[0]][black_loc[1]], image[white_loc[0]][white_loc[1]]

                bw = (gs > np.max(gs)/2).astype(int)
                bw_1D = np.reshape(bw, bw.shape[0]*bw.shape[1])
                rle_bw, t = bw_rle_encode(bw_1D)
                

                print('size:', len(rle_bw)*rle_bw.itemsize, 'type:', t)
                dec *= 2
                if len(rle_bw)*rle_bw.itemsize < TRANS_SIZE:
                    break

            return {'rle_image': rle_bw, 'black': black, 'white': white, 'shape': image.shape, 'dec_shape': dec_shape, 'type': t}, BLACK_WHITE

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
        pkt_bytes = None

        comp_header = struct.pack(COMP_TYPE, comp_type)
        if comp_type == NO_COMPRESSION:
            img_shape = img_comp.shape
            img_1D = np.reshape(img_comp, img_shape[0]*img_shape[1]*img_shape[2])
            img_bytes = img_1D.tobytes()

            header = struct.pack(NO_COMP_HEADER, *img_shape)
            pkt_bytes = comp_header + header + img_bytes
        elif comp_type == DECIMATE:
            img = img_comp['decimated_image']
            img_shape = img.shape
            img_1D = np.reshape(img, img_shape[0]*img_shape[1]*img_shape[2])
            img_bytes = img_1D.tobytes()

            header = struct.pack(DECIMATE_HEADER, *(img_comp['shape'] + img.shape))
            pkt_bytes = comp_header + header + img_bytes
        elif comp_type == BLACK_WHITE:
            rle = img_comp['rle_image']
            rle_bytes = rle.tobytes()

            black, white = img_comp['black'], img_comp['white']
            orig_shape = img_comp['shape']
            dec_shape = img_comp['dec_shape']

            type_bit = 0
            if img_comp['type'] == np.uint16:
                type_bit = 1
            elif img_comp['type'] == np.uint8:
                type_bit = 2

            header = struct.pack(BW_HEADER, *(tuple(black) + tuple(white) + orig_shape + dec_shape + (type_bit, )))
            pkt_bytes = comp_header + header + rle_bytes
            print('rle:', len(rle))
            print('rle_bytes:', len(rle_bytes))
            print('header:', len(header))
            print('pkt_bytes:', len(pkt_bytes))

        return pkt_bytes


    def send_bits(self, bytes):
        """
        Sends bytes through Baofeng.
        """
        print('Sending', len(bytes), 'bytes.')
        if self._lp_mode:
            send_queue.put(bytes)
        else:
            from transmit import send_bytes
            send_bytes(bytes)


class Receiver(object):

    def __init__(self, lp_mode):
        self._lp_mode = lp_mode

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
        pkt = None

        pkt_bytes = bits #.tobytes()

        comp_type_bytes, pkt_bytes = pkt_bytes[:COMP_TYPE_SIZE], pkt_bytes[COMP_TYPE_SIZE:]
        comp_type = struct.unpack(COMP_TYPE, comp_type_bytes)[0]
        if comp_type == NO_COMPRESSION:
            noc_header, pkt_bytes = pkt_bytes[:NO_COMP_HEADER_SIZE], pkt_bytes[NO_COMP_HEADER_SIZE:]
            noc_struct = struct.unpack(NO_COMP_HEADER, noc_header)
            shape = noc_struct

            img_1D = np.fromstring(pkt_bytes, dtype=np.uint8)
            img = np.reshape(img_1D, shape)

            pkt = NO_COMPRESSION, img
        elif comp_type == DECIMATE:
            dec_header, pkt_bytes = pkt_bytes[:DECIMATE_HEADER_SIZE], pkt_bytes[DECIMATE_HEADER_SIZE:]
            dec_struct = struct.unpack(DECIMATE_HEADER, dec_header)

            orig_shape, dec_shape = dec_struct[:3], dec_struct [3:]

            img_1D = np.fromstring(pkt_bytes, dtype=np.uint8)
            decimated = np.reshape(img_1D, dec_shape)
            data = {'decimated_image': decimated, 'shape': orig_shape}

            pkt = DECIMATE, data
        elif comp_type == BLACK_WHITE:
            bw_header, pkt_bytes = pkt_bytes[:BW_HEADER_SIZE], pkt_bytes[BW_HEADER_SIZE:]
            bw_struct = struct.unpack(BW_HEADER, bw_header)

            black, white, orig_shape, dec_shape, type_bit = bw_struct[:3], bw_struct[3:6], bw_struct[6:9], bw_struct[9:12], bw_struct[12]
            t = int
            if type_bit == 1:
                t = np.uint16
            elif type_bit == 2:
                t = np.uint8

            rle = np.fromstring(pkt_bytes, dtype=t)
            return BLACK_WHITE, {'rle_image': rle, 'black': black, 'white': white, 'shape': orig_shape, 'dec_shape': dec_shape}

        return pkt

    def packet_to_img(self, pkt):
        """
        Turns the packet into an image.
        """
        image = None

        comp_type, image_comp = pkt
        if comp_type == NO_COMPRESSION:
            image = image_comp
        elif comp_type == DECIMATE:
            orig_shape = image_comp['shape']

            bytes_original = 3 * orig_shape[0] * orig_shape[1]
            down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))

            upsampled = imresize(image_comp['decimated_image'], orig_shape, interp='nearest').astype(np.uint8)
            h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
            upsample_xy = convolve_image(upsampled, h_lpf)

            image = upsample_xy
        elif comp_type == BLACK_WHITE:
            rle_bw, black, white, orig_shape, dec_shape = image_comp['rle_image'], image_comp['black'], image_comp['white'], image_comp['shape'], image_comp['dec_shape']

            bw_1D = bw_rle_decode(rle_bw)
            bw = np.reshape(bw_1D, (dec_shape[0], dec_shape[1]))
            bw_inv = (bw < 1).astype(np.uint8)
            red_w, green_w, blue_w = bw*white[0], bw*white[1], bw*white[2]
            red_b, green_b, blue_b = bw_inv*black[0], bw_inv*black[1], bw_inv*black[2]
            red, green, blue = red_w + red_b, green_w + green_b, blue_w + blue_b

            recon = np.zeros(dec_shape).astype(np.uint8)
            recon[:,:,0], recon[:,:,1], recon[:,:,2] = red, green, blue
            if dec_shape != orig_shape:
                recon = upsampled = imresize(recon, orig_shape, interp='nearest').astype(np.uint8)
            image = recon

        return image

    def read_bits(self):
        """
        Read bits from SDR
        """
        if self._lp_mode:
            try:
                return send_queue.get(timeout=75)
            except QueueEmptyError:
                return None
        else:
            from transmit import receive_bytes
            print('Listening for bytes...')
            b = receive_bytes()
            print('Got bytes.')
            return b

        

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

    encoded, t = np.array(encoded), int
    if max(encoded) < 256:
        encoded, t = encoded.astype(np.uint8), np.uint8
    elif max(encoded) < 2**16:
        encoded, t = encoded.astype(np.uint16), np.uint16

    return encoded, t

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

