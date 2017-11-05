import sys
sys.path.append('./lib/DeepFlow_release2.0')
sys.path.append('./lib/deepmatching_1.2.2_c++')

from deepmatching import deepmatching
from deepflow2 import deepflow2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

im1 = cv2.resize(cv2.imread('sintel1.png'), (200, 200))
im2 = cv2.resize(cv2.imread('sintel2.png'), (200, 200))
im1 = np.array(im1)
im2 = np.array(im2)
matches = deepmatching(im1, im2)
flow = deepflow2(im1, im2, matches, '-sintel')

# Showing the animation
for i in range(20):
    for j in range(20):
        x = i * 10
        y = j * 10
        fx = flow[x, y, 0]
        fy = flow[x, y, 1]
        cv2.line(im1, (x,y), (int(x+fx), int(y+fy)), (255, 0, 0))
        cv2.circle(im1, (x,y), 1, (0, 255, 0), -1)
        cv2.line(im2, (x,y), (int(x-fx), int(y-fy)), (255, 0, 0))
        cv2.circle(im2, (x,y), 1, (0, 255, 0), -1)

def redraw_fn(f, axes):
    img = [im1, im2][f % 2]
    if not redraw_fn.initialized:
        redraw_fn.im = axes.imshow(img, animated=True)
        redraw_fn.initialized = True
    else:
        redraw_fn.im.set_array(img)
    time.sleep(0.5)

redraw_fn.initialized = False
from videofig import videofig
videofig(100, redraw_fn, play_fps=30)
