import cv2

def getflowbetnframe(im1, im2):
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    flow = 0
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, flow, pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0)
    return flow
