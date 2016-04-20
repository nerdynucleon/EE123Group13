from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError
from scipy import ndimage
import cv2

NO_COMPRESSION = 0
CARTOON_THRESHOLD = 50

send_queue = Queue()

class Transmitter(object):

    def __init__(self):
        pass

    def transmit(self, image):
        """
        Compresses and transmits image in 75 seconds
        """

        img_comp, comp_type = self.compress_image(image)
        bits = self.packet_to_bits(self.to_packet(img_comp, comp_type))

        Process(target=self.send_bits, args=(bits, )).start()


    def compress_image(self, image):
        """
        Compresses the image so it will fit in a packet. Returns an integer
        for the type of compression it uses.
        """

        return image, NO_COMPRESSION


    def to_packet(self, img_comp, comp_type):
        """
        Turns the compressed image into a bitstream packet
        """

        return img_comp


    def packet_to_bits(self, packet):
        """
        Turns the packet into bits to be sent over the radio.
        """

        return packet


    def send_bits(self, bits):
        """
        Sends bits through Baofeng.
        """

        send_queue.put(bits)

    def detect_img_type(self, image):
        """
        Decides whether image is cartoon or natural, influencing the
        compression type. Returns bool (1 is natural, 0 is cartoon).
        """
        image_smoothed = []
        cv.Smooth(image, image_smoothed, smoothtype=CV_BILATERAL, param1=3, param2=0, param3=0, param4=0)

        #the bigger the difference the less cartoon like it is
        cartoon_index = mean(image - image_smoothed)
        return cartoon_index > CARTOON_THRESHOLD:


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

        return bits


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
