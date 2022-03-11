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
import torch

############################################################

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

#############################################################

def face_detect(images):
    print('Length of Image list:',len(images))
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = 5#args.face_det_batch_size
    
    #while 1:
    predictions=[]
    for i in tqdm(range(0, len(images), batch_size)):
        #print('i:',i)
        im = np.array(images[i:i + batch_size])
            #image1 = torch.FloatTensor(im).unsqueeze(0)
        start_time = time.time()
        predictions.extend(detector.get_detections_for_batch(np.asarray(im)))
        print("Running time in seconds",(time.time() - start_time))
        
    results = []
    pady1, pady2, padx1, padx2 = 0,10,0,0
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    #print('Length of result:', len(results))
    del detector
    return results 

#############################################################

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print('Inside datagen')
    face_det_results = face_detect(frames)
    #print(type(face_det_results))
        
    for i, m in enumerate(mels):
        print('i:',i)
        idx = i%len(frames)
        #print('idx:',idx)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy() 
        #print(len(coords))

        face = cv2.resize(face, (img_size, img_size))
        #face = np.ones((96,96,3))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords) 

        if len(img_batch) >= 1: #args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            #img_batch, mel_batch, frame_batch = [], [], []
            
    # if len(img_batch) > 0:
    #     img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

    #     img_masked = img_batch.copy()
    #     img_masked[:, img_size//2:] = 0

    #     img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    #     mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    #     yield img_batch, mel_batch, frame_batch, coords_batch


#############################################################

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

checkpoint_path = '/content/drive/MyDrive/Wav2Lip-master/checkpoints/wav2lip_gan.pth'

mel_step_size=16
video_root = '/content/drive/MyDrive/Wav2Lip-master/data_root/main/Test4/video'
audio_root = '/content/drive/MyDrive/Wav2Lip-master/data_root/main/Test4/audio'
audio_file = glob(os.path.join(audio_root, '*.wav'))[0]
video_file = glob(os.path.join(video_root, '*.jpg'))[0]
video_stream = cv2.VideoCapture(video_file) 
fps = video_stream.get(cv2.CAP_PROP_FPS)
fps=25 ## For image
print('fps:',fps)
im = cv2.imread(video_file) ## For image
full_frames = []
all_frames=[]
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break
    
    all_frames.append(frame)

print ("Number of frames available for inference: "+str(len(full_frames)))

wav = audio.load_wav(audio_file, 16000)
mel = audio.melspectrogram(wav)
print(mel.shape)

mel_chunks = []
mel_idx_multiplier = 80./fps 
i = 0

while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

#mel_chunks=mel_chunks[50:] #201
#full_frames = full_frames[:20]
print("Length of mel chunks: {}".format(len(mel_chunks)))
#frames1=[]
#for i in range(len(mel_chunks)):
 # frames1.append(full_frames[0])
### ## For image ## ###
for i in range(len(mel_chunks)):
    full_frames.append(im)
    #full_frames.append(all_frames[0])
print ("Number of frames available for inference: "+str(len(full_frames)))
#full_frames = full_frames[:len(mel_chunks)]
gen = datagen(full_frames.copy(), mel_chunks)
#print('data')

# for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
#     print('Inside data generation loop')
#     print(coords)

for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
    print('Inside data generation loop')
    if i == 0:
            model = load_model(checkpoint_path)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('/content/drive/MyDrive/Wav2Lip-master/temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
            
    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        pred = model(mel_batch, img_batch)
        print("Running time in seconds",(time.time() - start_time))
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
    for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            #print('Before resizing:', p.shape)
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            #print('After resizing:', p.shape)

            f[y1:y2, x1:x2] = p
            #f = cv2.resize(f.astype(np.uint8), (256, 256), interpolation = cv2.INTER_CUBIC)
            p = np.array(f,dtype=np.uint8)
            out.write(p)

out.release()