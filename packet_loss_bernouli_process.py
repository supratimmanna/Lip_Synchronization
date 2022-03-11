import numpy as np
import audio
import os
import sys
from glob import glob 
from os import listdir, path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.io.wavfile import write
import math

#created a bernoulli class
class bernoulli():
    
    def __init__(self, p):
        self.p = p
    #probability mass function
    def pmf(self, x):
        f = self.p**x*(1-self.p)**(1-x)
        return f
    
    def mean(self):
        return self.p
    
    def var(self):
        return self.p*(1-self.p)
    
    def std(self):
        return bernoulli.var(self.p)**(1/2)
    
    # Generate random variables
    def rvs(self, size=1):
        rvs = np.array([])
        for i in range(0,size):
            if np.random.rand() <= self.p:
                a=0
                rvs = np.append(rvs,a)
            else:
                a=1
                rvs = np.append(rvs,a)
        return rvs
####################################################

sr = 16000
n_fft = 800
hop_length = 200
win_length=800
n_mels=80
#dir = 'data_root\\main\\Test4\\audio'
dir = 'LRS2_audio\\audio'
af = glob(os.path.join(dir, '*.wav'))[0]

wav = audio.load_wav(af, sr)
#wav=wav[:sr*5]
t = int(wav.shape[0]/sr)
print("Total number of sample in the original audio:", wav.shape[0])
print("Duration of audio in seconds:", t)
sample_per_frame = 16000/25

packet_size_in_ms = 10
packet_size_in_sample = (sr/1000)*packet_size_in_ms

total_packets = math.ceil(wav.shape[0]/packet_size_in_sample)

### Create audio packets (each packet has 160 samples) ###############
packets = []
for i in range(total_packets):
    start = i*160
    end = start+160
    packets.append(wav[start:end])
packets = np.array(packets, dtype=object)
#########################################################

packet_loss_proba=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for j,p in enumerate(packet_loss_proba):
    bern = bernoulli(p)
    lost_packet_idx = bern.rvs(size=total_packets)
    num_lost_packets = np.count_nonzero(lost_packet_idx==0)
    print("Total Number of Lost Packets:", num_lost_packets) 
    ## Perform packet loss effect ###################
    audio_with_lost_packet = np.multiply(packets, lost_packet_idx)
    lost_audio = []
    for i in range(len(audio_with_lost_packet)):
        each_packet = audio_with_lost_packet[i]
        for sample in each_packet:
            lost_audio.append(sample)
    lost_audio = np.array(lost_audio)
    ## Save the audio with lost packet
    lost_audio_path = os.path.join(dir,'lost_audio_'+str(int(p*100))+ '%.wav')
    print(j)
    write(lost_audio_path, 16000, lost_audio)
