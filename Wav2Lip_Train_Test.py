from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import re

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

data_root = 'lrs2_preprocessed\\train'

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split, data_root):
        self.all_videos = get_image_list(data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y
        
logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

###################################
def train(device, model, train_data_loader, optimizer, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
 
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            g = model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            prog_bar.set_description('L1: {}, Sync Loss: {}'.format(running_l1_loss / (step + 1),
                                                                    running_sync_loss / (step + 1)))

        global_epoch += 1
###################################


train_data=Dataset('train', data_root) 
train_data_loader = data_utils.DataLoader(
        train_data, batch_size=hparams.batch_size, shuffle=True,
        num_workers=2) 

device = torch.device("cuda" if use_cuda else "cpu")
model = Wav2Lip().to(device)
print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

train(device, model, train_data_loader, optimizer, nepochs=2)
######################################################
        
# train_data=Dataset('train', data_root) 
# idx = random.randint(0, len(train_data.all_videos) - 1)
# vidname = train_data.all_videos[idx]
# img_names = list(glob(join(vidname, '*.jpg')))

# img_names.sort(key=lambda f: int(re.sub('\D', '', f)))

# img_name = random.choice(img_names)
# wrong_img_name = random.choice(img_names)
# while wrong_img_name == img_name:
#     wrong_img_name = random.choice(img_names)
    
# window_fnames = train_data.get_window(img_name)
# wrong_window_fnames = train_data.get_window(wrong_img_name) 
# window = train_data.read_window(window_fnames)
# wrong_window = train_data.read_window(wrong_window_fnames)

# wavpath = join(vidname, "audio.wav")
# wav = audio.load_wav(wavpath, hparams.sample_rate)
# orig_melt = audio.melspectrogram(wav).T
# melt = train_data.crop_audio_window(orig_melt.copy(), img_name)
# indiv_melst = train_data.get_segmented_mels(orig_melt.copy(), img_name)
# window = train_data.prepare_window(window)
# y = window.copy()
# window[:, :, window.shape[2]//2:] = 0.
# wrong_window = train_data.prepare_window(wrong_window)
# x = np.concatenate([window, wrong_window], axis=0)

# x = torch.FloatTensor(x)
# melt = torch.FloatTensor(melt.T).unsqueeze(0)
# indiv_melst = torch.FloatTensor(indiv_melst).unsqueeze(1)
# y = torch.FloatTensor(y)

# indiv_melst = torch.FloatTensor(indiv_melst).unsqueeze(0)
# x = torch.FloatTensor(x).unsqueeze(0)
# y = torch.FloatTensor(y).unsqueeze(0)
# wav2lip = Wav2Lip().to(device)
# optimizer = optim.Adam([p for p in wav2lip.parameters() if p.requires_grad],
#                            lr=hparams.initial_learning_rate)

# wav2lip.train()
# optimizer.zero_grad()

# g = wav2lip(indiv_melst, x)

# melt = torch.FloatTensor(melt).unsqueeze(0)
# for i in range(5):
#     melt = torch.cat([melt, melt])
#     g = torch.cat([g, g])
#     y = torch.cat([y, y])
    
# sync_loss = get_sync_loss(melt, g)

# y = y.to(device)
# l1loss = recon_loss(g, y)
# loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
# loss.backward()
# optimizer.step()


# # for i in range(len(img_name)):
# #     img_names.append(glob(join(vidname, str(i), '.jpg')))
    
        
