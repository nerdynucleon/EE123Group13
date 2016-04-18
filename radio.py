from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue
from Queue import Empty as QueueEmptyError

send_queue = Queue()

class Transmitter(object):

    def __init__(self):
        pass

    def transmit(self, image):
        """
        Compresses and transmits image in 75 seconds
        """


        def add_image(queue, image):
            queue.put(image)

        Process(target=add_image, args=(send_queue, image)).start()


class Receiver(object):

    def __init__(self):
        pass

    def receive(self):
        """
        Recieves images and returns them.
        """

        try:
            return send_queue.get(timeout=75)
        except QueueEmptyError:
            return None
        
