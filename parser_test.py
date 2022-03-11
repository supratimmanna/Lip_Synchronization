from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=True)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

args = parser.parse_args()

def main():
    print('Given fps is:', args.fps)
    # image = cv2.imread('0.jpg')
    # h = image.shape[0]
    # w = image.shape[1]
    # print('Original Image shape:',image.shape)
    # rf = args.resize_factor
    # new_dim = (int(h*rf), int(w*rf))
    # resized = cv2.resize(image, new_dim)
    # print('Resized Image shape:', resized.shape)

if __name__ == '__main__':
	main()