import subprocess
import os
import sys
from glob import glob 
from os import listdir, path
import moviepy.editor as mp

dir = 'data_root\\main\\train'
#filelist = glob(os.path.join(dir, '*.avi'))
filelist = glob('*.avi')
video_file=filelist[0]

video = mp.VideoFileClip(video_file)
video.audio.write_audiofile(r"output.mp3")