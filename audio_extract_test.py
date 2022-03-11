import subprocess
import os
import sys
from glob import glob 
from os import listdir, path

#template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    template = 'ffmpeg -y -i{} -strict -2 {}'
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    
    
dir = 'data_root\\main\\Train'
filelist = glob(os.path.join(dir, '*.mp4'))

#for a in filelist:
a=filelist[0]
processed_dir = 'lrs2_preprocessed'
filename, ext = os.path.splitext(a)
vidname = os.path.basename(a).split('.')[0]
dirname = a.split('\\')[-2] 
sub_fulldir = path.join(processed_dir, dirname, vidname)
os.makedirs(sub_fulldir, exist_ok=True)
fulldir = path.join(processed_dir, dirname, vidname,'audio')
subprocess.call(["ffmpeg", "-y", "-i", a, f"{filename}.{'wav'}"], 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
#convert_video_to_audio_ffmpeg(a)