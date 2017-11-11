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

def write_list(framelist, basedir):
    for frame in framelist:
        vidcap = cv2.VideoCapture(frame[0])
        (image, flow) = get_flow(vidcap, frame[1])
        impath = frame[0].replace('UCF-101', basedir + '/final') + frame[1] + ".png"
        flowpath = frame[0].replace('UCF-101', basedir + '/flow') + frame[1] + ".flow"
        safe_dir_create(impath)
        safe_dir_create(flowpath)
        cv2.imwrite(impath, image)
        pickle.dump(flow, open( flowpath, "wb" ) )

trainsize = math.floor(len(framelist) * 0.8)
write_list(framelist[:trainsize], 'UCF-preprocessed/training')
write_list(framelist[trainsize+1:], 'UCF-preprocessed/test')
