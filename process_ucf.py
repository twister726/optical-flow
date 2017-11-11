from get_frames import get_random_frames
from get_flow import get_flow
import cv2
import os
import shutil
import random
import math
import pickle

framelist = get_random_frames()
random.shuffle(framelist)

print(framelist)

def safe_dir_create(path):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

try:
    shutil.rmtree('./datasets/UCF-processed')
    os.makedirs('./datasets/UCF-processed')
except:
    pass

def write_frame(vidcap, basedir, path, frame):
    (image, flow) = get_flow(vidcap, frame)
    impath = path.replace('UCF-101', basedir + '/final') + str(frame) + ".png"
    flowpath = path.replace('UCF-101', basedir + '/flow') + str(frame) + ".flow"
    safe_dir_create(impath)
    safe_dir_create(flowpath)
    cv2.imwrite(impath, image)
    pickle.dump(flow, open( flowpath, "wb" ) )

def write_list(framelist, basedir):
    i = 0
    length = len(framelist)
    for (path, frames) in framelist:
        print('Video %d out of %d\n' % (i, length))
        vidcap = cv2.VideoCapture(path)
        for frameno in frames:
            write_frame(vidcap, basedir, path, frameno)

trainsize = int(math.floor(len(framelist) * 0.8))
write_list(framelist[:trainsize], 'UCF-processed/training')
write_list(framelist[trainsize+1:], 'UCF-processed/test')
