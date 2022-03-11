from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import audio

import torch
from torch import nn
from torch.nn import functional as F
import math
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
import librosa
from models.conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()
        
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])
        
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        self.face_decoder_blocks = nn.ModuleList([
           nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

           nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
           Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

           nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
           Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
           Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

           nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
           Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
           Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

           nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
           Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
           Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

           nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
           Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
           Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

           nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
           Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
           Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
           nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
           nn.Sigmoid()) 
    
    def forward(self, audio_sequences, face):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)
        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        
        face_sequences = torch.cat([face[:, :, i] for i in range(face.size(2))], dim=0)
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)
        x = torch.split(x, 1, dim=0) # [(B, C, H, W)]
        outputs = torch.stack(x, dim=2)
        return audio_embedding, feats, outputs
    
    def forward_face(self, face):
        face_sequences = torch.cat([face[:, :, i] for i in range(face.size(2))], dim=0)
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        #feats = np.asarray(feats)
        return feats
    
    
    
    
    
    
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

device = torch.device("cuda" if use_cuda else "cpu")

def crop_audio_window(spec, start_frame):
        start_frame_num = start_frame
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]
    
    
def get_segmented_mels(spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = start_frame
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels
    
wav, sr = librosa.load(librosa.ex('trumpet'), hparams.sample_rate)
orig_mel = audio.melspectrogram(wav).T
mel = crop_audio_window(orig_mel.copy(), 12)
indiv_mels = get_segmented_mels(orig_mel.copy(), 12)

mel = torch.FloatTensor(mel.T).unsqueeze(0) 
indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(0)

mel_save = mel
mel = torch.FloatTensor(mel).unsqueeze(0)

mel = mel.to(device)
indiv_mels = indiv_mels.to(device)
xx=indiv_mels

# # B = indiv_mels.size(0)

# xx=[]
# xx.append(indiv_mels)
# xx.append(indiv_mels)
# xx = np.array(xx)
# xx = torch.FloatTensor(xx).unsqueeze(2)

# xx = xx.to(device)

xxx = torch.cat([xx[:, i] for i in range(5)], dim=0)

wav2lip = Wav2Lip()

aa=np.ones((5,96,96,3))
aa = np.asarray(aa)

aa = np.transpose(aa, (3, 0, 1, 2))

xf = np.concatenate([aa, aa], axis=0)
xf = torch.FloatTensor(xf)
xf = torch.FloatTensor(xf).unsqueeze(0)
#aa = torch.FloatTensor(aa)
#xf = np.concatenate([aa, aa], axis=0)

audio_embedding, face_embedding, g = wav2lip.forward(indiv_mels, xf)
g_save = g


class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
syncnet = SyncNet_color()
g = g[:, :, :, g.size(3)//2:]
g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
for i in range(5):
    mel = torch.cat([mel, mel])
    g = torch.cat([g, g])
# #mel = torch.FloatTensor(mel).unsqueeze(1)
a, f = syncnet.forward(g, mel)
# y = torch.ones(g.size(0), 1).float().to(device)

# d = nn.functional.cosine_similarity(a, f)
# logloss = nn.BCELoss()
# loss = logloss(d.unsqueeze(1), y)
