import sys
sys.path.append('./lib/DeepFlow_release2.0')
sys.path.append('./lib/deepmatching_1.2.2_c++')
sys.path.append('./get_flow')

from deepmatching import deepmatching
from deepflow2 import deepflow2
import numpy as np
import cv2
import matplotlib.pyplot as plt

from show_flow import show_flow

def getflowbetnframe(im1, im2):
    matches = deepmatching(im1, im2)
    flow = deepflow2(im1, im2, matches, '-sintel')
    outflow = np.empty([20, 20, 2])
    for i in range(20):
        for j in range(20):
            outflow[i,j,0] = np.average(flow[10*i:10*i+10, 10*j:10*j+10, 0])
            outflow[i,j,1] = np.average(flow[10*i:10*i+10, 10*j:10*j+10, 1])
    return outflow

def process_frame(image):
    image = cv2.resize(image, (200, 200))
    return np.array(image)

def get_flow(vidcap, frame, display=False):
    vidcap.set(1, frame-1)
    success, image = vidcap.read(1)
    if not success:
        raise ValueError('Frame not present in video')
    image = process_frame(image)
    succ_frames = []
    for i in range(10):
        # vidcap.set(1, frame + i - 1)
        success, tmpimage = vidcap.read(1)
        if not success:
            continue
        succ_frames.append(process_frame(tmpimage))
    print(len(succ_frames))
    flows = [getflowbetnframe(image, tmpframe) for tmpframe in succ_frames]
    avgflow = np.zeros([20, 20, 2], dtype=np.float)
    for flow in flows:
        avgflow += flow
    avgflow = avgflow / len(succ_frames)
    show_flow(image, succ_frames, flows)
    return avgflow