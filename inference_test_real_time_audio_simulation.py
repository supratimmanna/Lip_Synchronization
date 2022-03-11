from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import time

############################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

###################################################################


img_size = 96
batch_size=1

checkpoint_path = '/content/drive/MyDrive/Wav2Lip-mastercheckpoints/wav2lip_gan.pth'

mel_step_size=16
video_root = '/content/drive/MyDrive/Wav2Lip-master/data_root/main/Test1/video'
audio_root = '/content/drive/MyDrive/Wav2Lip-master/data_root/main/Test1/audio'
audio_file = glob(os.path.join(audio_root, '*.wav'))[1]
video_file = glob(os.path.join(video_root, '*.mp4'))[0]
video_stream = cv2.VideoCapture(video_file)
fps = video_stream.get(cv2.CAP_PROP_FPS)
print(fps)

full_frames = []
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break
    
    full_frames.append(frame)

#print ("Number of frames available for inference: "+str(len(full_frames)))


wav = audio.load_wav(audio_file, 16000)
mel_chunks = []

for i in range(219):
    p = i*640
    wav_cut = wav[p:p+640]
    mel = audio.melspectrogram(wav_cut)
    mel_chunks.append(mel)

#mel_chunks=mel_chunks[:5] #201
#full_frames = full_frames[:10]
print("Length of mel chunks: {}".format(len(mel_chunks)))
#full_frames = full_frames[:len(mel_chunks)]

######### Real time simultion for audio ###########################



##################### Real time simulationn #######################

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                        flip_input=False, device='cpu')
for idx in range(len(mel_chunks)):
    image = full_frames[idx]
    im = torch.FloatTensor(image).unsqueeze(0)
    start_time = time.time()
    predictions=(detector.get_detections_for_batch(np.asarray(im)))
    print("Running time in seconds",(time.time() - start_time))
    
    results = []
    pady1, pady2, padx1, padx2 = 0,10,0,0
    for rect, image in zip(predictions, im):
        #print('rect:', rect)
        #print(image.shape[0], image.shape[1])
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
    
        results.append([x1, y1, x2, y2])
    
    boxes = np.array(results)
    #boxes = get_smoothened_boxes(boxes, T=5)
    face_det_results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(im, boxes)]

    img_batch=[]
    mel_batch=[]
    frame_batch=[]
    coords_batch=[]

    m = mel_chunks[idx]
    frame_to_save = full_frames[idx].copy()
    face, coords = face_det_results[0].copy() 
    print('coords:',coords)
    face = cv2.resize(np.array(face), (img_size, img_size))
#face = np.ones((96,96,3))
    
    img_batch.append(face)
    mel_batch.append(m)
    frame_batch.append(frame_to_save)
    coords_batch.append(coords) 

    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch) 

    img_masked = img_batch.copy()
    img_masked[:, img_size//2:] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    if idx==0:
        print('idx=0')
        model = load_model(checkpoint_path)

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        
    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

    with torch.no_grad():
        start_time = time.time()
        pred = model(mel_batch, img_batch)
        #print("Running time in seconds",(time.time() - start_time))
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

    for p, f, c in zip(pred, frame_batch, coords_batch):
        y1, y2, x1, x2 = c
        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
        f[y1:y2, x1:x2] = p
        out.write(f)
        
out.release()
################################################################