from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError
from scipy import signal
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import filters as im_filters
from scipy import fftpack, linalg
from PIL import Image
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
DECIMATE_HEADER = '7i'
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
        original_shape = image.shape
        if imtype == None:
            mod_image, imtype = self.determine_imtype(image)

        img_comp, comp_type  = self.compress_image(image, mod_image, imtype, original_shape)
        bits = self.to_packet_bits(img_comp, comp_type, imtype)

        Process(target=self.send_bits, args=(bits, )).start()


    def compress_image(self, original, image, imtype, original_shape):
        """
        Compresses the image so it will fit in a packet. Returns an integer
        for the type of compression it uses.
        """
        DWT_result = None
        # Finds the best DWT Compression that Fits
        DWT_thresh = 87 #XXX
        DWT_result_size = TRANS_SIZE + 1
        while DWT_result_size > TRANS_SIZE and DWT_thresh <= 127: #max values for int8
            DWT_result = haar_downsample(image, DWT_thresh)
            if imtype != COLOR:
                DWT_result_size = DWT_result[0].size * 2 + DWT_result[1].size
            else:
                DWT_result_size = DWT_result[0].size * 5/3 + DWT_result[1].size
            print('DWT Threshold: ' + str(DWT_thresh) + " DWT_Size: " + str(DWT_result_size))
            DWT_thresh += 1

        # Finds the best DCT Compression that Fits
        DCT_result = None
        DCT_thresh = 112 #XXX
        DCT_result_size = TRANS_SIZE + 1
        while DCT_result_size > TRANS_SIZE and DCT_thresh <= 127:
            DCT_result = dct_downsample(image, DCT_thresh)
            if imtype != COLOR:
                DCT_result_size = DCT_result[0].size * 2 + DCT_result[1].size
            else:
                DCT_result_size = DCT_result[0].size * 5/3 + DCT_result[1].size
            print('DCT Threshold: ' + str(DCT_thresh) + " DCT_Size: " + str(DCT_result_size))
            DCT_thresh += 1


        # Finds the best DECIMATE Compression that Fits
        DECIMATE_result = self.decimate_compression(image, imtype)
        #RLE_result = self.run_length_encoding_compression(image)
        NO_COMPRESSION_result = None

        DWT_PSNR = PSNR(original, haar_upsample(DWT_result, original_shape))
        DCT_PSNR = PSNR(original, dct_upsample(DCT_result, original_shape))
        DECIMATE_PSNR = PSNR(original, decimate_upsample(DECIMATE_result, imtype, original_shape))
        #RLE_PSNR = PSNR(image, self.run_length_encoding_decompression(RLE_result, imtype))
        NO_COMPRESSION_PSNR = PSNR(original, NO_COMPRESSION_result)

        psnr_arr = np.array([DWT_PSNR, DCT_PSNR, DECIMATE_PSNR, NO_COMPRESSION_PSNR])
        print(psnr_arr)
        result_arr = [({'dwt_indices' : DWT_result[0], 'dwt_values' : DWT_result[1], 'shape' : original_shape}, DWT), \
                      ({'dct_indices' : DCT_result[1], 'dct_values' : DCT_result[1], 'shape' : original_shape}, DCT), \
                      ({'decimated_image': DECIMATE_result, 'shape': original_shape}, DECIMATE), \
                      (image, NO_COMPRESSION)]
        return result_arr[0]
        #return result_arr[np.argmax(psnr_arr)]


    def run_length_encoding_compression(self, image):
        gs = np.average(image, axis=2)
        black_loc = np.unravel_index(gs.argmin(), gs.shape)
        white_loc = np.unravel_index(gs.argmax(), gs.shape)
        black, white = image[black_loc[0]][black_loc[1]], image[white_loc[0]][white_loc[1]]

        bw = (gs > np.max(gs)/2).astype(int)
        bw_1D = np.reshape(bw, bw.shape[0]*bw.shape[1])
        rle_bw = bw_rle_encode(bw_1D)
        return {'rle_image': rle_bw, 'black': black, 'white': white, 'shape': image.shape}

    def run_length_encoding_decompression(self, image, imtype):
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
        image = np.reshape(image[:,:,0], (image.shape[0], image.shape[1], 1))
        down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))
        down_sampling_factor = int(down_sampling_factor)
        h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
        image_lpf = convolve_image(image, h_lpf)
        decimated_xy = image_lpf[::down_sampling_factor, ::down_sampling_factor]
        return decimated_xy

    def to_packet_bits(self, img_comp, comp_type, imtype):
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

            header = struct.pack(DECIMATE_HEADER, *(img_comp['shape'] + img.shape + (imtype,)))
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


    def is_color_grayscale(self, img, size=40, mean_error_thresh=22):
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
        if mean_error <= mean_error_thresh:
            return 0
        else:
            return 1

    def determine_imtype(self, image):
        """
        Decide if image is grayscale, bw(binary), or color
        """

        if self.is_color_grayscale(image):
            imtype = COLOR
        elif self.is_black_white(image):
            imtype = BLACK_WHITE
            image = np.reshape(image[:,:,0], (image.shape[0], image.shape[1], 1))
        else:
            imtype = GRAYSCALE
            image = np.reshape(image[:,:,0], (image.shape[0], image.shape[1], 1))
        return image, imtype

    def is_black_white(self, image):
        """
        Returns 1 for bw, 0 for anything else
        """
        img = np.array(image, dtype=np.float)
        if((np.sum(img == 255) + np.sum(img == 0)) == img.size):
            return 1
        else:
            return 0


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

            orig_shape, dec_shape, imtype = dec_struct[:3], dec_struct [3:6], dec_struct[6]

            img_1D = np.fromstring(pkt_bytes, dtype=np.uint8)
            decimated = np.reshape(img_1D, dec_shape)
            data = {'decimated_image': decimated, 'shape': orig_shape, 'imtype' : imtype}

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
            imtype = image_comp['imtype']
            image_compressed = image_comp['decimated_image']
            image = decimate_upsample(image_compressed, imtype, orig_shape)
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

def decimate_upsample(image, imtype, orig_shape):
    bytes_original = orig_shape[0] * orig_shape[1]
    if imtype == COLOR:
        bytes_original *= 3
    else:
        image = np.concatenate((image,image,image),axis = 2)
    down_sampling_factor = np.ceil(np.sqrt(bytes_original / TRANS_SIZE))
    down_sampling_factor = int(down_sampling_factor)
    upsampled = imresize(image, orig_shape, interp='nearest').astype(np.uint8)
    h_lpf = gaussian_lpf(np.pi / down_sampling_factor)
    upsample_xy = convolve_image(upsampled, h_lpf)
    return upsample_xy

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
    layers = ()
    for i in range(3):
        layer_lpf = im_filters.convolve(image[:,:,i], filt) # signal.convolve2d(image[:,:,i], filt, mode='same')
        layers += (layer_lpf.astype(np.uint8), )
    
    return np.dstack(layers)

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

def PSNR(im_truth, im_test, maxval=255.):
    if im_test == None:
        return 0
    else:
        mse = np.linalg.norm(im_truth.astype(np.float64) - im_test.astype(np.float64))**2 / np.prod(np.shape(im_truth))
        psnr = 10 * np.log10(maxval**2 / mse)
        return psnr

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
    image_glob = img_out
    return compression_serialize(threshold, img_out)

def compression_serialize(threshold, img_transformed):
    img_transformed[np.absolute(img_transformed) < threshold] = 0
    values = img_transformed[img_transformed.nonzero()]
    indices = np.absolute(img_transformed) >= threshold
    indices0, indices1, indices2 = indices.nonzero()
    if img_transformed.shape[2] > 1:
        indices = np.vstack((indices0.astype(np.uint16), indices1.astype(np.uint16), indices2.astype(np.uint8)))
    else:
        indices = np.vstack((indices0.astype(np.uint16), indices1.astype(np.uint16)))
    return indices, values

def deserialize(indices, values, original_shape, N):
    padded_shape = (N * np.ceil(original_shape[0] / N), N * np.ceil(original_shape[1] / N), original_shape[2])
    image = np.zeros(np.array(padded_shape).astype(np.uint16))
    indicex = np.array(indices[0,:])
    indicey = np.array(indices[1,:])
    if indices.shape[0] > 2:
        indicez = np.array(indices[2,:])
    else:
        indicez = np.zeros((1,indices.shape[1])).astype(np.uint16)
    image[(indicex, indicey, indicez)] = values
    image = image.astype(np.int8)
    return image

def haar_upsample(indices_values, original_shape):
    N = 8
    # reconstruct image
    image = deserialize(indices_values[0], indices_values[1], original_shape, N)
    padded_shape = (N * np.ceil(original_shape[0] / N), N * np.ceil(original_shape[1] / N), original_shape[2])
    padded_shape = np.array(padded_shape).astype(np.uint16)
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
                img_out[i:i+N, j:j+N, k] = fftpack.dct(fftpack.dct(img[i:i+N, j:j+N, k].T, norm='ortho').T, norm='ortho')

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
    padded_shape = np.array(padded_shape).astype(np.uint16)
    img_out = np.zeros(padded_shape)
    haar = haar_matrix(N)

    for k in range(image.shape[2]):
        for i in range(0,image.shape[0], N):
            for j in range(0, image.shape[1], N):
                img_out[i:i+N, j:j+N, k] = fftpack.idct(fftpack.idct(image[i:i+N, j:j+N, k].T, norm='ortho').T, norm='ortho')

    return 4 * img_out[0:original_shape[0], 0:original_shape[1], 0:original_shape[2]]

def create_runs(length, value):
    runs = []
    while length > 255:
        runs.extend([255, value])
        length -= 255
    runs.extend([length, value])
    return runs

def rle_encode(array_1D):
    changes, = np.where(np.diff(array_1D) != 0)
    changes = np.concatenate(([0], changes + 1, [len(array_1D)]))
    rle = np.array([create_runs(b-a, array_1D[a]) for a, b in zip(changes[:-1], changes[1:])])
    rle_flat = []
    for l in rle:
        rle_flat.extend(l)
    return np.array(rle_flat, dtype=np.uint8)

def rle_decode(rle_enc):
    decoded = []
    for i in range(0, len(rle_enc), 2):
        run = [rle_enc[i+1]]*rle_enc[i]
        decoded.extend(run)
    return np.array(decoded, dtype=np.uint8)


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
