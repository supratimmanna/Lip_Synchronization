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


randomlist = []
for i in range(0,5):
    n = random.randint(1,30)
    randomlist.append(n)


sr = 16000
n_fft = 800
hop_length = 200
win_length=800
n_mels=80
dir = 'data_root\\main\\Test4\\audio'
af = glob(os.path.join(dir, '*.wav'))[0]

wav = audio.load_wav(af, sr)
#wav=wav[:sr*5]
t = int(wav.shape[0]/sr)
print(t)
sample_per_frame = 16000/25

packet_size_in_ms = 10
packet_size_in_sample = (sr/1000)*packet_size_in_ms

total_packets = math.ceil(wav.shape[0]/packet_size_in_sample)
wave=[]
count=0
for j in range(total_packets):
    if j%5==0 or j%6==0:
        #print(j)
        count+=1
        start = int(j*packet_size_in_sample)
        end = int(start + packet_size_in_sample)
        wav[start:end] = 0
#wav=wav[:16000]
#t = int(wav.shape[0]/sr) 
# noise = np.random.normal(0,0.1,len(wav))
# wav_noise = wav+noise
# #wav_noise = np.clip(wav_noise, -0.5, 0.5)
# noisy_audio_path = os.path.join(dir,'noisy_audio.wav')

# write(noisy_audio_path, 16000, wav_noise)

# delete = 4
# for i in range(t):
#     j=i*sr
#     start = j+1000
#     end = start+delete*1024
#     #print(start, end)
#     wav[start:end]=0
#     # start = j+6000
#     # end = start+delete*1024
#     # #print(start, end)
#     # wav[start:end]=0
#     start = j+8000
#     end = start+delete*1024
#     #print(start, end)
#     wav[start:end]=0
# END=[]
# for i in range(t):
#     j=i*sr
#     start = j+100
#     for f in range(25):
#         start = int(j + (f*640) + 100)
#         end = int(start+(sample_per_frame/1))
#         END.append(end)
#         wav[start:end]=0

lost_audio_path = os.path.join(dir,'lost_audio.wav')
write(lost_audio_path, 16000, wav)




orig_mel = audio.melspectrogram(wav)
om = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length, n_mels=n_mels)




fig, ax = plt.subplots()
S_dB = librosa.power_to_db(om, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                          y_axis='mel', sr=sr,
                          fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')