# Showing the animation
import cv2
import time

from videofig import videofig

def show_flow(image, succ_frames, flows):
    global frames
    frames = [ image ] + succ_frames
    for ind in range(len(succ_frames)):
        im = succ_frames[ind]
        flow = flows[ind]
        for i in range(20):
            for j in range(20):
                x = i * 10 + 5
                y = j * 10 + 5
                fx = flow[i, j, 0]
                fy = flow[i, j, 1]
                cv2.circle(im, (x,y), 1, (0, 255, 0), -1)
                cv2.line(im, (x,y), (int(x-fx), int(y-fy)), (255, 0, 0))
    videofig(100, redraw_fn, play_fps=30)

def redraw_fn(f, axes):
    global frames
    img = frames[f % len(frames)]
    if not redraw_fn.initialized:
        redraw_fn.im = axes.imshow(img, animated=True)
        redraw_fn.initialized = True
    else:
        redraw_fn.im.set_array(img)
    time.sleep(0.5)

redraw_fn.initialized = False
