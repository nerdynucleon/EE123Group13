from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError

NO_COMPRESSION = 0

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
        
