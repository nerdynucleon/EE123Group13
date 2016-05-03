from __future__ import division, print_function
from rtlsdr import RtlSdr, limit_time
from matplotlib import pyplot as plt
import numpy as np

count = 0

@limit_time(10)
def print_points(points, sdr):
    sdr.cancel_read_async()
    global count
    # plt.plot(abs(points))
    # plt.show()
    count += 1
    if count >= 50:
        print('Finished inputting', count)
        # sdr.cancel_read_async()

fs_sdr = 240000
fc =    443.63e6 # set your frequency!
ppm_start =   51 # set estimate ppm
gain =  20 # set gain

sdr = RtlSdr()
sdr.sample_rate = fs_sdr    # sampling rate
sdr.gain = gain
sdr.center_freq = fc
sdr.set_freq_correction(ppm_start)

try:
    sdr.read_samples_async(print_points)
    print('test')
except IOError:
    sdr.close()
    raise

