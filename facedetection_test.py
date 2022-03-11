from glob import glob
import os
from os import listdir, path
import sys
import multiprocessing as mp
import cv2 
import numpy as np
#import ffmpeg
import face_detection
import numpy as np
import argparse, os, cv2, traceback, subprocess
from hparams import hparams as hp
import torch

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
 	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
 							before running this script!')
                            
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

dir = 'data_root\\main\\train'
filelist = glob(os.path.join(dir, '*.mp4'))

#for a in filelist:
a=filelist[0]
#vidname = os.path.basename(a).split('.')[0]
#dirname = a.split('\\')[-2]
processed_dir = 'lrs2_preprocessed'
#fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False)
fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cpu')

image = cv2.imread('0.jpg')
image1 = torch.FloatTensor(image).unsqueeze(0)
preds = fa.get_detections_for_batch(np.asarray(image1))
for j, f in enumerate(preds):
    x1, y1, x2, y2 = f
cv2.imwrite(path.join('pred'+'{}.jpg'.format(100)), image[y1:y2, x1:x2])
#aa = batches[3]
#aa = np.asarray(aa)
# for i,image in enumerate(aa):
#     image1 = torch.FloatTensor(image).unsqueeze(0)
#     preds = fa.get_detections_for_batch(np.asarray(image1))
#     for j, f in enumerate(preds):
#         x1, y1, x2, y2 = f
#     cv2.imwrite(path.join('pred', '{}.jpg'.format(100)), image[y1:y2, x1:x2])