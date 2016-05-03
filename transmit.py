import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import Queue
import threading,time
import sys

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import ones
from scipy import signal
from scipy import integrate

import threading,time
import multiprocessing

from numpy import mean
from numpy import power
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
import bitarray
from  scipy.io.wavfile import read as wavread
import serial
import ax25
from fractions import gcd

import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *


# function to compute least common multipler
def lcm(numbers):
    return reduce(lambda x, y: (x*y)/gcd(x,y), numbers, 1)


def play_audio( Q,ctrlQ ,p, fs , dev, ser="", keydelay=0.1):
    # play_audio plays audio with sampling rate = fs
    # Q - A queue object from which to play
    # ctrlQ - A queue object for ending the thread
    # p   - pyAudio object
    # fs  - sampling rate
    # dev - device number
    # ser - pyserial device to key the radio
    # keydelay - delay after keying the radio
    #
    #
    # There are two ways to end the thread: 
    #    1 - send "EOT" through  the control queue. This is used to terminate the thread on demand
    #    2 - send "EOT" through the data queue. This is used to terminate the thread when data is done. 
    #
    # You can also key the radio either through the data queu and the control queue
    
    
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs), output=True, output_device_index=dev)
    # play audio
    while (1):
        if not ctrlQ.empty():
            
            # control queue 
            ctrlmd = ctrlQ.get()
            if ctrlmd is "EOT"  :
                    ostream.stop_stream()
                    ostream.close()
                    print("Closed  play thread")
                    return;
            elif (ctrlmd is "KEYOFF"  and ser!=""):
                ser.setDTR(0)
                #print("keyoff\n")
            elif (ctrlmd is "KEYON" and ser!=""):
                ser.setDTR(1)  # key PTT
                #print("keyon\n")
                time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up 
        
        data = Q.get()

        if (data is "EOT") :
            ostream.stop_stream()
            ostream.close()
            print("Closed  play thread")
            return;
        elif (data is "KEYOFF"  and ser!=""):
            ser.setDTR(0)
            #print("keyoff\n")
        elif (data is "KEYON" and ser!=""):
            ser.setDTR(1)  # key PTT
            #print("keyon\n")
            time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up
        else:
            try:
                ostream.write( data.astype(np.float32).tostring() )
            except:
                print("Exception", data)
                raise
                break
            
def record_audio( queue,ctrlQ, p, fs ,dev,chunk=1024):
    # record_audio records audio with sampling rate = fs
    # queue - output data queue
    # p     - pyAudio object
    # fs    - sampling rate
    # dev   - device number 
    # chunk - chunks of samples at a time default 1024
    #
    # Example:
    # fs = 44100
    # Q = Queue.queue()
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # record_audio( Q, p, fs, 1) # 
    # p.terminate() # terminate pyAudio
    
   
    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];
    while (1):
        if not ctrlQ.empty():
            ctrlmd = ctrlQ.get()          
            if ctrlmd is "EOT"  :
                istream.stop_stream()
                istream.close()
                print("Closed  record thread")
                return;
        try:  # when the pyaudio object is distroyed stops
            data_str = istream.read(chunk) # read a chunk of data
        except:
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        queue.put( data_flt ) # append to list


dusb_in = 2
dusb_out = 2
din = 0
dout = 1

s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')


class TNCaprs:
    
    def __init__(self, fs = 48000.0, Abuffer = 1024, Nchunks=43):
        
        #  Implementation of an afsk1200 TNC. 
        #
        #  The TNC processes a `Abuffer` long buffers, till `Nchunks` number of buffers are collected into a large one.
        #  This is because python is able to more efficiently process larger buffers than smaller ones.
        #  Then, the resulting large buffer is demodulated, sampled and packets extracted.
        #
        # Inputs:
        #    fs  - sampling rate
        #   TBW  -  TBW of the demodulator filters
        #   Abuffer - Input audio buffers from Pyaudio
        #   Nchunks - Number of audio buffers to collect before processing
        #   plla    - agressivness parameter of the PLL
        
        
        ## compute sizes based on inputs
        self.TBW = 2.0   # TBW for the demod filters
        self.N = (int(fs/1200*self.TBW)//2)*2+1   # length of the filters for demod
        self.fs = fs     # sampling rate   
        self.BW = self.TBW/(1.0*self.N/fs)      # BW of filter based on TBW
        self.Abuffer = Abuffer             # size of audio buffer
        self.Nchunks = Nchunks             # number of audio buffers to collect
        self.Nbuffer = Abuffer*Nchunks+self.N*3-3         # length of the large buffer for processing
        self.Ns = 1.0*fs/1200 # samples per symbol
        
        ## state variables for the modulator
        self.prev_ph = 0  # previous phase to maintain continuous phase when recalling the function
        
        ##  Generate Filters for the demodulator
        self.h_lp = signal.firwin(self.N,self.BW/fs*1.0,window='hanning')
        self.h_lpp = signal.firwin(self.N,self.BW*2*1.2/fs,window='hanning')
        self.h_space = self.h_lp*exp(1j*2*pi*(2200)*r_[-self.N/2:self.N/2]/fs)
        self.h_mark = self.h_lp*exp(1j*2*pi*(1200)*r_[-self.N/2:self.N/2]/fs)
        self.h_bp = signal.firwin(self.N,self.BW/fs*2.2,window='hanning')*exp(1j*2*pi*1700*r_[-self.N/2:self.N/2]/fs)


        ## PLL state variables  -- so conntinuity between buffers is preserved
        self.dpll = np.round(2.0**32 / self.Ns).astype(int32)    # PLL step
        self.pll =  0                # PLL counter
        self.ppll = -self.dpll       # PLL counter previous value -- to detect overflow
        self.plla = 0.74             # PLL agressivness (small more agressive)
        

        ## state variable to NRZI2NRZ
        self.NRZIprevBit = True  
        
        ## State variables for findPackets
        self.state='search'   # state variable:  'search' or 'pkt'
        self.pktcounter = 0   # counts the length of a packet
        self.packet = bitarray.bitarray([0,1,1,1,1,1,1,0])   # current packet being collected
        self.bitpointer = 0   # poiter to advance the search beyond what was already searched in the previous buffer

        ## State variables for processBuffer
        self.buff = zeros(self.Nbuffer)   # large overlapp-save buffer
        self.chunk_count = 0              # chunk counter
        self.oldbits = bitarray.bitarray([0,0,0,0,0,0,0])    # bits from end of prev buffer to be copied to beginning of new
        self.Npackets = 0                 # packet counter
        
        
    
    
    def NRZ2NRZI(self,NRZ, prevBit = True):
        NRZI = NRZ.copy() 
        for n in range(0,len(NRZ)):
            if NRZ[n] :
                NRZI[n] = prevBit
            else:
                NRZI[n] = not(prevBit)
            prevBit = NRZI[n]
        return NRZI

    def NRZI2NRZ(self, NRZI):  
        NRZ = NRZI.copy() 
    
        for n in range(0,len(NRZI)):
            NRZ[n] = NRZI[n] == self.NRZIprevBit
            self.NRZIprevBit = NRZI[n]
    
        return NRZ
    
    def modulate(self,bits):
    # the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them, sampled at 44100Hz
    #  Inputs:
    #         bits  - bitarray of bits
    #         fs    - sampling rate
    # Outputs:
    #         sig    -  returns afsk1200 modulated signal
    
        fss = lcm((1200,self.fs))
        deci = fss/self.fs
    
        Nb = fss/1200
        nb = len(bits)
        NRZ = ones((nb,Nb))
        for n in range(0,nb):
            if bits[n]:
                NRZ[n,:]=-NRZ[n,:]
    
        freq = 1700 + 500*NRZ.ravel()
        ph = self.prev_ph + 2.0*pi*integrate.cumtrapz(freq)/fss
        sig = cos(ph[::deci])
        
        return sig 
    
    def modulatPacket(self, callsign, digi, dest, info, preflags=80, postflags=80 ):
        
        # given callsign, digipath, dest, info, number of pre-flags and post-flags the function contructs
        # an appropriate aprs packet, then converts them to NRZI and calls `modulate` to afsk 1200 modulate the packet. 
        
        packet = ax25.UI(destination=dest,source=callsign, info=info, digipeaters=digi.split(b','),)
        prefix = bitarray.bitarray(np.tile([0,1,1,1,1,1,1,0],(preflags,)).tolist())
        suffix = bitarray.bitarray(np.tile([0,1,1,1,1,1,1,0],(postflags,)).tolist())
        sig = self.modulate(self.NRZ2NRZI(prefix + packet.unparse()+suffix))

        return sig
    
    

    def demod(self, buff):

    
        SIG = signal.fftconvolve(buff,self.h_bp,mode='valid')
        mark = signal.fftconvolve(SIG,self.h_mark,mode='valid')
        space = signal.fftconvolve(SIG,self.h_space,mode='valid')
        NRZ = abs(mark)-abs(space)
        NRZ = signal.fftconvolve(NRZ,self.h_lpp,mode='valid')
        return NRZ


    
    def PLL(self, NRZa):        
        idx = zeros(len(NRZa)//int(self.Ns)*2)   # allocate space to save indexes        
        c = 0
        
        for n in range(1,len(NRZa)):
            if (self.pll < 0) and (self.ppll >0):
                idx[c] = n
                c = c+1
        
            if (NRZa[n] >= 0) !=  (NRZa[n-1] >=0):
                self.pll = int32(self.pll*self.plla)
            
        
            self.ppll = self.pll
            self.pll = int32(self.pll+ self.dpll)
    
        return idx[:c].astype(int32) 
    
   

    def findPackets(self,bits):
        # function take a bitarray and looks for AX.25 packets in it. 
        # It implements a 2-state machine of searching for flag or collecting packets
        flg = bitarray.bitarray([0,1,1,1,1,1,1,0])
        packets = []
        n = self.bitpointer
        
        # Loop over bits
        while (n < len(bits)-7) :
            # default state is searching for packets
            if self.state is 'search':
                # look for 1111110, because can't be sure if the first zero is decoded
                # well if the packet is not padded.
                if bits[n:n+7] == flg[1:]:
                    # flag detected, so switch state to collecting bits in a packet
                    # start by copying the flag to the packet
                    # start counter to count the number of bits in the packet
                    self.state = 'pkt'
                    self.packet=flg.copy()
                    self.pktcounter = 8
                    # Advance to the end of the flag
                    n = n + 7
                else:
                    # flag was not found, advance by 1
                    n = n + 1            
        
            # state is to collect packet data. 
            elif self.state is 'pkt':
                # Check if we reached a flag by comparing with 0111111
                # 6 times ones is not allowed in a packet, hence it must be a flag (if there's no error)
                if bits[n:n+7] == flg[:7]:
                    # Flag detected, check if packet is longer than some minimum
                    if self.pktcounter > 200:
                        # End of packet reached! append packet to list and switch to searching state
                        # We don't advance pointer since this our packet might have been
                        # flase detection and this flag could be the beginning of a real packet
                        self.state = 'search'
                        self.packet.extend(flg)
                        packets.append(self.packet.copy())
                    else:
                        # packet is too short! false alarm. Keep searching 
                        # We don't advance pointer since this this flag could be the beginning of a real packet
                        self.state = 'search'
                # No flag, so collect the bit and add to the packet
                else:
                    # check if packet is too long... if so, must be false alarm
                    if self.pktcounter < 2680:
                        # Not a false alarm, collect the bit and advance pointer        
                        self.packet.append(bits[n])
                        self.pktcounter = self.pktcounter + 1
                        n = n + 1
                    else:  #runaway packet
                        #runaway packet, switch state to searching, and advance pointer
                        self.state = 'search'
                        n = n + 1
        
        self.bitpointer = n-(len(bits)-7) 
        return packets

    
    # function to generate a checksum for validating packets
    def genfcs(self,bits):
        # Generates a checksum from packet bits
        fcs = ax25.FCS()
        for bit in bits:
            fcs.update_bit(bit)
    
        digest = bitarray.bitarray(endian="little")
        digest.frombytes(fcs.digest())

        return digest




    # function to parse packet bits to information
    def decodeAX25(self,bits, deepsearch=False):
        ax = ax25.AX25()
        ax.info = "bad packet"
    
    
        bitsu = ax25.bit_unstuff(bits[8:-8])
    
        
        foundPacket = False
        if (self.genfcs(bitsu[:-16]).tobytes() == bitsu[-16:].tobytes()):
                foundPacket = True
        elif deepsearch: 
            tbits = bits[8:-8]
            for n in range(0,len(tbits)):
                tbits[n] = not tbits[n]
                if (self.genfcs(bitsu[:-16]).tobytes() == bitsu[-16:].tobytes()):
                    foundPacket = True
                    print("Success deep search")
                    break
                tbits[n] = not tbits[n]
        
        if foundPacket == False:
            return ax
        
        
                  
    
        bytes = bitsu.tobytes()
        ax.destination = ax.callsign_decode(bitsu[:56])
        source = ax.callsign_decode(bitsu[56:112])
        if source[-1].isdigit() and source[-1]!="0":
            ax.source = b"".join((source[:-1],'-',source[-1]))
        else:
            ax.source = source[:-1]
    
        digilen=0    
    
        if bytes[14]=='\x03' and bytes[15]=='\xf0':
            digilen = 0
        else:
            for n in range(14,len(bytes)-1):
                if ord(bytes[n]) & 1:
                    digilen = (n-14)+1
                    break

        #    if digilen > 56:
        #        return ax
        ax.digipeaters =  ax.callsign_decode(bitsu[112:112+digilen*8])
        ax.info = bitsu[112+digilen*8+16:-16].tobytes()
    
        return ax

    def processBuffer(self, buff_in):
        
        # function processes an audio buffer. It collect several small into a large one
        # Then it demodulates and finds packets.
        #
        # The function operates as overlapp and save
        # The function returns packets when they become available. Otherwise, returns empty list
        
        N = self.N
        NN = N*3-3
        
        
        Nchunks = self.Nchunks
        Abuffer = self.Abuffer
        fs = self.fs
        Ns = self.Ns
        
        validPackets=[]
        packets=[]
        NRZI=[]
        idx = []
        bits = []
        
        # Fill in buffer at the right plave
        self.buff[NN+self.chunk_count*Abuffer:NN+(self.chunk_count+1)*Abuffer] = buff_in.copy()
        self.chunk_count = self.chunk_count + 1
        
        
        # number of chunk reached -- process large buffer
        if self.chunk_count == Nchunks:
            # Demodulate to get NRZI
            NRZI = self.demod(self.buff)
            # compute sampling points, using PLL
            idx = self.PLL(NRZI)
            # Sample and make a decision based on threshold
            bits = bitarray.bitarray((NRZI[idx]>0).tolist())
            # In case that buffer is too small raise an error -- must have at least 7 bits worth
            if len(bits) < 7:
                raise ValueError('number of bits too small for buffer')
            
            # concatenate end of previous buffer to current one
            bits = self.oldbits + self.NRZI2NRZ(bits)
            
            # store end of bit buffer to next buffer
            self.oldbits = bits[-7:].copy()
            
            # look for packets
            packets = self.findPackets(bits)
            
            # Copy end of sample buffer to the beginning of the next (overlapp and save)
            self.buff[:NN] = self.buff[-NN:].copy()
            
            # reset chunk counter
            self.chunk_count = 0
            
            # checksum test for all detected packets
            for n in range(0,len(packets)):
                if len(packets[n]) > 200: 
                    ax = self.decodeAX25(packets[n])
                    if ax.info != 'bad packet':
                        validPackets.append(ax)
                        
            
        return validPackets


callsign = "KM6KBA"
fs = 48000
modem = TNCaprs(fs ,Abuffer = 1024,Nchunks = 12)


def receive_bytes():
    Qin = Queue.Queue()
    cQin = Queue.Queue()
    p = pyaudio.PyAudio()
    fs_usb = p.get_device_info_by_index(2)['defaultSampleRate']
    t_rec = threading.Thread(target=record_audio, args=(Qin, cQin, p, fs_usb, dusb_in))

    t_rec.start()
    npack = 0
    state = 0
    f1 = None
    while(1):
        tmp = Qin.get()
        packets  = modem.processBuffer(tmp)
        for ax in packets: 
            npack = npack + 1
            print((str(npack)+")",str(ax)))
            if state == 0 and ax.destination[:5]=="BEGIN":
                f1 = "" # open("rec_"+ax.info,"wb")
                state = 1
            elif state == 1 and ax.destination[:3] == "END": 
                state = 2  
                break
            elif state == 1:
                f1 += ax.info # f1.write(ax.info)
        if state == 2 :
            break
    time.sleep(10)
    cQin.put("EOT")
    return f1

def send_bytes(data):
    Qout = Queue.Queue()
    cQout = Queue.Queue()
    p = pyaudio.PyAudio()
    fs_usb = p.get_device_info_by_index(2)['defaultSampleRate']

    s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
    s.setDTR(0)
    # t_play = threading.Thread(target=play_audio, args=(Qout, cQout, p, fs_usb, dusb_out, s))

    npp = 0
    Qout.put(np.zeros(fs_usb/2))
    Qout.put("KEYON")
    Qout.put(np.zeros(fs_usb/2))
    tmp = modem.modulatPacket(callsign, "", "BEGIN", 'test.tiff', preflags=80, postflags=2 ) 
    Qout.put(0.1*tmp)
    while(1):
        chunk_size = min(len(data), 256)
        bytes_s, data = data[:chunk_size], data[chunk_size:] # f.read(256)    
        tmp = modem.modulatPacket(callsign, "", str(npp), bytes_s, preflags=4, postflags=2 )    
        Qout.put(0.1*tmp)
        npp = npp+1
        if len(data) == 0:
            break
    tmp = modem.modulatPacket(callsign, "", "END", "This is the end of transmission", preflags=2, postflags=80 )
    Qout.put(0.1*tmp)
    Qout.put("KEYOFF")
    Qout.put("EOT")
    play_audio(Qout, cQout, p, fs, dusb_out, s)
    # t_play.start()
    cQout.put("EOT")
        
