from glob import glob
import os
import time
from os import listdir, path
import sys
import multiprocessing as mp
import cv2 
import numpy as np
import subprocess
import traceback
#import ffmpeg
import face_detection
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp
import torch

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

if not path.isfile('/content/drive/MyDrive/Wav2Lip-master/face_detection/detection/sfd/s3fd.pth'):
 	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
 							before running this script!')
                            
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

dir = '/content/drive/MyDrive/Wav2Lip-master/data_root/main/Train'
filelist = glob(os.path.join(dir, '*.mp4'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#for a in filelist:
a=filelist[1]
#vidname = os.path.basename(a).split('.')[0]
#dirname = a.split('\\')[-2]
processed_dir = '/content/drive/MyDrive/Wav2Lip-master/lrs2_preprocessed'
#fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False)
fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

def process_video_file_face(vfile, processed_dir, batch_size):
    video_stream= cv2.VideoCapture(vfile)
    frames = []
    
    while 1:
         still_reading, frame = video_stream.read()
         if not still_reading:
             video_stream.release()
             break
         frames.append(frame)
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    
    fulldir = path.join(processed_dir, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)
    
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    aa = batches[2]
    aa = np.asarray(aa)
    print(aa.shape)
    preds = fa.get_detections_for_batch(np.asarray(aa))
    print(preds)
    i=-1
    for image in aa:
        i+=1
        image1 = torch.FloatTensor(image).unsqueeze(0)
        print('Image:',image.shape, 'Image1:', image1.shape)
        start_time = time.time()
        preds = fa.get_detections_for_batch(np.asarray(image1))
        print("Running time in seconds",(time.time() - start_time))
        for j, f in enumerate(preds):
            x1, y1, x2, y2 = f
        cv2.imwrite(path.join(fulldir,'{}.jpg'.format(i)), image[y1:y2, x1:x2])
        
        
def process_video_file_face_batch(vfile, processed_dir, batch_size):
    video_stream= cv2.VideoCapture(vfile)
    frames = []
    
    while 1:
         still_reading, frame = video_stream.read()
         if not still_reading:
             video_stream.release()
             break
         frames.append(frame)
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    print('Total number of Frames:',len(frames))
    fulldir = path.join(processed_dir, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)
    
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    i=-1
    print('Total number of batches:',len(batches))
    for images in batches:
        #print(i)
        #aa = np.asarray(aa)
        images = np.asarray(images)
        #print(images[0].shape)
        start_time = time.time()
        preds = fa.get_detections_for_batch(images)
        print("Running time in seconds",(time.time() - start_time))
        for j, f in enumerate(preds):
            i+=1
            #print(i,j)
            x1, y1, x2, y2 = f
            #print(x1)
            cv2.imwrite(path.join(fulldir,'{}.jpg'.format(i)), images[j][y1:y2, x1:x2])
    #aa = batches[2]
    #start_time = time.time()
    #preds = fa.get_detections_for_batch(np.asarray(aa))
    #print("Running time in seconds",(time.time() - start_time))
    # i=-1
    # for image in aa:
    #     i+=1
    #     image1 = torch.FloatTensor(image).unsqueeze(0)
    #     print('Image:',image.shape, 'Image1:', image1.shape)
    #     start_time = time.time()
    #     preds = fa.get_detections_for_batch(np.asarray(image1))
    #     print("Running time in seconds",(time.time() - start_time))
    #     for j, f in enumerate(preds):
    #         x1, y1, x2, y2 = f
    #     cv2.imwrite(path.join(fulldir,'{}.jpg'.format(i)), image[y1:y2, x1:x2])
     

def process_video_file_face_main(vfile, processed_dir, batch_size):
 	video_stream = cv2.VideoCapture(vfile)
 	
 	frames = []
 	while 1:
          still_reading, frame = video_stream.read()
          if not still_reading:
              video_stream.release()
              break
          frames.append(frame)
 	
 	vidname = os.path.basename(vfile).split('.')[0]
 	dirname = vfile.split('\\')[-2]

 	fulldir = path.join(processed_dir, dirname, vidname)
 	os.makedirs(fulldir, exist_ok=True)

 	batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

 	i = -1
 	for fb in batches:
          preds = fa.get_detections_for_batch(np.asarray(fb))
          for j, f in enumerate(fb):
              i += 1
              if f is None:
                  continue
              x1, y1, x2, y2 = f
              cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])    



def process_video_file(vfile, processed_dir, batch_size):
 	video_stream = cv2.VideoCapture(vfile)
 	
 	frames = []
 	while 1:
          still_reading, frame = video_stream.read()
          if not still_reading:
              video_stream.release()
              break
          frames.append(frame)
 	
 	vidname = os.path.basename(vfile).split('.')[0]
 	dirname = vfile.split('\\')[-2]

 	fulldir = path.join(processed_dir, dirname, vidname)
 	os.makedirs(fulldir, exist_ok=True)

 	batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

 	i = -1
 	for fb in batches:
          for j, f in enumerate(fb):
              i += 1
              if f is None:
                  continue
              cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j])



def process_audio_file(vfile, processed_dir):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('\\')[-2]
    sub_fulldir = path.join(processed_dir, dirname, vidname)
    os.makedirs(sub_fulldir, exist_ok=True)
    fulldir = path.join(processed_dir, dirname, vidname,'audio')
    subprocess.call(["ffmpeg", "-y", "-i", a, f"{fulldir}.{'wav'}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return vidname

process_video_file_face_batch(a, processed_dir, 5)    
#for aa in filelist[0]:
    #process_video_file(aa, processed_dir, 32)
    #process_audio_file(aa, processed_dir) 
    