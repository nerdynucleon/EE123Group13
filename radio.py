from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError
from scipy import signal
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import filters as im_filters
from transmit import send_bytes, receive_bytes
import numpy as np
import encoding as enc
import struct
import bitarray
import matplotlib.pyplot as plt

TRANS_SIZE = 7500

# Compression types
NO_COMPRESSION = 0
DECIMATE = 1
RUN_LENGTH_ENCODING = 2
DWT = 3
DCT = 4

# Image Types
COLOR = 0
GRAYSCALE = 1
BLACK_WHITE = 2

send_queue = Queue()


# Struct Formats
COMP_TYPE = 'b'
COMP_TYPE_SIZE = struct.calcsize(COMP_TYPE)
NO_COMP_HEADER = '3i'
NO_COMP_HEADER_SIZE = struct.calcsize(NO_COMP_HEADER)
DECIMATE_HEADER = '6i'
DECIMATE_HEADER_SIZE = struct.calcsize(DECIMATE_HEADER)
BW_HEADER = '9i'
BW_HEADER_SIZE = struct.calcsize(BW_HEADER)

class Transmitter(object):

    def __init__(self):
        pass

    def transmit(self, image, imtype=None):
        """
        Compresses and transmits image in 75 seconds
        """
        if imtype == None:
            imtype = self.determine_imtype(image)

        img_comp, comp_type = self.compress_image(image, imtype)
        bits = self.to_packet_bits(img_comp, comp_type)

        Process(target=self.send_bits, args=(bits, )).start()


    def compress_image(self, image, imtype):
        """
        Compresses the image so it will fit in a packet. Returns an integer
        for the type of compression it uses.
        """
        DWT_result = None
        # Finds the best DWT Compression that Fits
        DWT_thresh = 0
        while DWT_result[0].size + DWT_result[1].size > TRANS_SIZE and DWT_thresh <= 127: #max values for int8
            DWT_result = haar_downsample(image, DWT_thresh)
            DWT_thresh += 1
        # Finds the best DCT Compression that Fits
        DCT_result = None
        DCT_thresh = 0
        while DCT_result[0].size + DCT_result[1].size > TRANS_SIZE and DCT_thresh <= 127:
            DCT_result = dct_downsample(image, DCT_thresh)
            DCT_thresh += 1
        # Finds the best DECIMATE Compression that Fits
        DECIMATE_result = self.decimate_compression(image, imtype)
        RLE_result = self.run_length_encoding_compression(image)
        NO_COMPRESSION_result = None

        DWT_PSNR = PSNR(image, haar_upsample(DWT_result, image.shape))
        DCT_PSNR = PSNR(image, dct_upsample(DCT_result, image.shape))
        DECIMATE_PSNR = PSNR(image, decimate_upsample(DECIMATE_result, imtype, image.shape))
        RLE_PSNR = PSNR(image, self.run_length_encoding_decompression(RLE_result))
        NO_COMPRESSION_PSNR = PSNR(image, NO_COMPRESSION_result)

        psnr_arr = np.array([DWT_PSNR, DCT_PSNR, DECIMATE_PSNR, RLE_PSNR, NO_COMPRESSION_PSNR])
        result_arr = [({'dwt_indices' : DWT_result[0], 'dwt_values' : DWT_result[1], 'shape' : image.shape}, DWT), \
                      ({'dct_indices' : DCT_result[1], 'dct_values' : DCT_result[1], 'shape' : image.shape}, DCT), \
                      ({'decimated_image': decimated_xy, 'shape': image.shape}, DECIMATE), \
                      (RLE_result, RUN_LENGTH_ENCODING), \
                      (image, NO_COMPRESSION)]

        return result_arr[np.argmax(psnr_arr)]


    def run_length_encoding_compression(self, image):
        gs = np.average(image, axis=2)
        black_loc = np.unravel_index(gs.argmin(), gs.shape)
        white_loc = np.unravel_index(gs.argmax(), gs.shape)
        black, white = image[black_loc[0]][black_loc[1]], image[white_loc[0]][white_loc[1]]

        bw = (gs > np.max(gs)/2).astype(int)
        bw_1D = np.reshape(bw, bw.shape[0]*bw.shape[1])
        rle_bw = bw_rle_encode(bw_1D)
        return {'rle_image': rle_bw, 'black': black, 'white': white, 'shape': image.shape}

    def run_length_encoding_decompression(self, image):
        rle_bw, black, white, orig_shape = image['rle_image'], image['black'], image['white'], image['shape']

        bw_1D = bw_rle_decode(rle_bw)
        bw = np.reshape(bw_1D, (orig_shape[0], orig_shape[1]))
        bw_inv = (bw < 1).astype(np.uint8)
        red_w, green_w, blue_w = bw*white[0], bw*white[1], bw*white[2]
        red_b, green_b, blue_b = bw_inv*black[0], bw_inv*black[1], bw_inv*black[2]
        red, green, blue = red_w + red_b, green_w + green_b, blue_w + blue_b

        recon = np.zeros(orig_shape).astype(np.uint8)
        recon[:,:,0], recon[:,:,1], recon[:,:,2] = red, green, blue

    def decimate_compression(self, image, imtype):
        bytes_original = image.shape[0] * image.shape[1]
        if imtype == COLOR:
            bytes_original *= 3
        else:
            image = image[:,:,0]
        down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))
        h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
        image_lpf = convolve_image(image, h_lpf)
        decimated_xy = image_lpf[::down_sampling_factor, ::down_sampling_factor]
        return decimated_xy

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

            header = struct.pack(BW_HEADER, *(tuple(black) + tuple(white) + orig_shape))
            pkt_bytes = comp_header + header + rle_bytes

        bits = bitarray.bitarray()
        bits.frombytes(pkt_bytes)
        return pkt_bytes


    def send_bits(self, bits):
        """
        Sends bits through Baofeng.
        """
        print('Sending', len(bits), 'bits.')
        # send_queue.put(bits)
        send_bytes(bits)


    def is_color_grayscale(img, size=40, mean_error_thresh=22):
        """
        Returns 1 for color, 0 for grayscale (may include bw)
        """

        #use pil to resize to common/small shape, no need to go through hella pixels
        img_pil = Image.fromarray(img)
        img_thumb = img_pil.resize((size,size))

        #initialize vars, bias helps to detect monochromatic images (similar to grayscale)
        sum_error = 0
        bias = [0,0,0]

        #bias calc
        # bias = ImageStat.Stat(img_thumb).mean[:3]
        # bias = [b - sum(bias)/3 for b in bias ]
        for px in img_thumb.getdata():
            pixel_mean = sum(px)/3
            sum_error += sum((px[i] - pixel_mean - bias[i])**2 for i in [0,1,2])
        mean_error = float(sum_error)/(size**2)
        if mean_error <= mean_error_thresh
            return 0
        else:
            return 1

    def determine_imtype(self, image):
        """
        Decide if image is grayscale, bw(binary), or color
        """

        if is_color_grayscale(image):
            #color
            imtype = COLOR
        elif is_black_white:
            imtype = BLACK_WHITE
        else:
            imtype = GRAYSCALE

    def is_black_white(self, image):
        """
        Returns 1 for bw, 0 for anything else
        """
        img = np.array(image, dtype=np.float)
        if((np.sum(img == 255) + np.sum(img == 0)) == img.size):
            return 1
        else:
            return 0
    def decimate_upsample(self, image, imtype, orig_shape):
        bytes_original = orig_shape[0] * orig_shape[1]
        if imtype == COLOR:
            bytes_original *= 3
        down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))
        upsampled = imresize(image_comp['decimated_image'], orig_shape, interp='nearest').astype(np.uint8)
        h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
        upsample_xy = convolve_image(upsampled, h_lpf)
        return upsample_xy

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
        pkt = None

        pkt_bytes = bits.tobytes()

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

            black, white, orig_shape = bw_struct[:3], bw_struct[3:6], bw_struct[6:]
            rle = np.fromstring(pkt_bytes, dtype=int)
            return BLACK_WHITE, {'rle_image': rle, 'black': black, 'white': white, 'shape': orig_shape}

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
            rle_bw, black, white, orig_shape = image_comp['rle_image'], image_comp['black'], image_comp['white'], image_comp['shape']

            bw_1D = bw_rle_decode(rle_bw)
            bw = np.reshape(bw_1D, (orig_shape[0], orig_shape[1]))
            bw_inv = (bw < 1).astype(np.uint8)
            red_w, green_w, blue_w = bw*white[0], bw*white[1], bw*white[2]
            red_b, green_b, blue_b = bw_inv*black[0], bw_inv*black[1], bw_inv*black[2]
            red, green, blue = red_w + red_b, green_w + green_b, blue_w + blue_b

            recon = np.zeros(orig_shape).astype(np.uint8)
            recon[:,:,0], recon[:,:,1], recon[:,:,2] = red, green, blue

            image = recon

        return image

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

    return np.array(encoded).astype(int)

def bw_rle_decode(bw_rle):
    decoded = []
    curr_val = 0
    for run in bw_rle:
        decoded.extend([curr_val]*run)
        curr_val = 0 if curr_val else 1  # flip curr_val
    return np.array(decoded).astype(np.uint8)

## Shitty Alex

def PSNR(im_truth, im_test, maxval=255.):
    if im_test == None:
        return 0
    else:
        mse = np.linalg.norm(im_truth.astype(np.float64) - im_test.astype(np.float64))**2 / np.prod(np.shape(im_truth))
    return 10 * np.log10(maxval**2 / mse)

# http://fourier.eng.hmc.edu/e161/lectures/Haar/haar.html
def haar_matrix(N):
    if(not np.log2(N).is_integer()):
        print("Invalid Haar Matrix Size")
        raise Exception
    else:
        H = np.ones((1,N))
        for k in range(1,N):
            p = max(0, np.floor(np.log2(k)))
            q = k - (2.0**p) + 1
            H_pos_k_upper =  np.array((q - 1.0)/(2.0**p) <= np.linspace(0,1,N,False))
            H_pos_k_lower =  np.array( np.linspace(0,1,N,endpoint=False) < (q - 0.5)/(2.0**p))
            H_pos_k =  np.logical_and(H_pos_k_upper, H_pos_k_lower)
            H_neg_k_upper =  np.array((q - 0.5)/(2.0**p) <= np.linspace(0,1,N,endpoint=False))
            H_neg_k_lower =  np.array(np.linspace(0,1,N,endpoint=False) < (q)/(2.0**p))
            H_neg_k = np.logical_and(H_neg_k_upper, H_neg_k_lower)
            H_k = np.array(H_pos_k, dtype=np.int8) - np.array(H_neg_k, dtype=np.int8)
            H_k = H_k
            H = np.vstack((H,H_k))

    return linalg.orth(H)

def haar_downsample(image, threshold):
    N = 8
    haar = haar_matrix(N)

    img = image
    pad_0 = image.shape[0] % N
    if pad_0 != 0:
        pad_0 = N - pad_0
        pad_0 = np.zeros((pad_0, image.shape[1], image.shape[2]))
        img = np.vstack((img, pad_0))

    pad_1 = img.shape[1] % N
    if pad_1 != 0:
        pad_1 = N - pad_1
        pad_1 = np.zeros((img.shape[0], pad_1, image.shape[2]))
        img = np.hstack((img, pad_1))

    img_out = np.zeros(img.shape)

    for k in range(image.shape[2]):
        for i in range(0,img.shape[0], N):
            for j in range(0, img.shape[1], N):
                img_out[i:i+N, j:j+N, k] = np.dot(np.dot(haar,img[i:i+N, j:j+N, k]), np.transpose(haar))

    img_out /= 4
    img_out = img_out.astype(np.int8)
    global image_glob
    image_glob = img_out
    return compression_serialize(threshold, img_out)

def compression_serialize(threshold, img_transformed):
    img_transformed[np.absolute(img_transformed) < threshold] = 0
    values = img_transformed[img_transformed.nonzero()]
    indices = np.absolute(img_transformed) != 0
    indices = np.packbits(indices)
    return indices, values

def deserialize(indices, values, original_shape, N):
    padded_shape = (N * np.ceil(original_shape[0] / N), N * np.ceil(original_shape[1] / N), original_shape[2])
    image = np.unpackbits(indices)
    image = np.reshape(image, padded_shape)
    image = image.astype(np.int8)
    image[image.nonzero()] = values
    return image
def haar_upsample(indices_values, original_shape):
    N = 8
    # reconstruct image
    image = deserialize(indices_values[0], indices_values[1], original_shape, N)
    padded_shape = (N * np.ceil(original_shape[0] / N), N * np.ceil(original_shape[1] / N), original_shape[2])
    img_out = np.zeros(padded_shape)
    haar = haar_matrix(N)

    for k in range(image.shape[2]):
        for i in range(0,image.shape[0], N):
            for j in range(0, image.shape[1], N):
                img_out[i:i+N, j:j+N, k] = np.dot(np.dot(np.transpose(haar),image[i:i+N, j:j+N, k]), haar)

    return 4 * img_out[0:original_shape[0], 0:original_shape[1], 0:original_shape[2]]

def dct_downsample(image, threshold):
    N = 8
    haar = haar_matrix(N)

    img = image
    pad_0 = image.shape[0] % N
    if pad_0 != 0:
        pad_0 = N - pad_0
        pad_0 = np.zeros((pad_0, image.shape[1], image.shape[2]))
        img = np.vstack((img, pad_0))

    pad_1 = img.shape[1] % N
    if pad_1 != 0:
        pad_1 = N - pad_1
        pad_1 = np.zeros((img.shape[0], pad_1, image.shape[2]))
        img = np.hstack((img, pad_1))

    img_out = np.zeros(img.shape)

    for k in range(image.shape[2]):
        for i in range(0,img.shape[0], N):
            for j in range(0, img.shape[1], N):
                img_out[i:i+N, j:j+N, k] = ftpack.dct(fftpack.dct(img[i:i+N, j:j+N, k].T, norm='ortho').T, norm='ortho'))

    img_out /= 4
    img_out = img_out.astype(np.int8)
    global image_glob
    image_glob = img_out
    return compression_serialize(threshold, img_out)

def dct_upsample(indices_values, original_shape):
    N = 8
    # reconstruct image
    image = deserialize(indices_values[0], indices_values[1], original_shape, N)
    padded_shape = (N * np.ceil(original_shape[0] / N), N * np.ceil(original_shape[1] / N), original_shape[2])
    img_out = np.zeros(padded_shape)
    haar = haar_matrix(N)

    for k in range(image.shape[2]):
        for i in range(0,image.shape[0], N):
            for j in range(0, image.shape[1], N):
                img_out[i:i+N, j:j+N, k] = fftpack.idct(fftpack.idct(image[i:i+N, j:j+N, k].T, norm='ortho').T, norm='ortho')

    return 4 * img_out[0:original_shape[0], 0:original_shape[1], 0:original_shape[2]]
