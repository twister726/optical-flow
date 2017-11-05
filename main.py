# Sample code for get flow as of now
# To be replaced by preprocessing pipeline

import cv2

from get_flow import get_flow

videoFile = "./datasets/UCF-101/ThrowDiscus/v_ThrowDiscus_g01_c02.avi"
vidcap = cv2.VideoCapture(videoFile)

frameNo = 100
# Find flow for 20th frame
get_flow(vidcap, frameNo, display=True)
