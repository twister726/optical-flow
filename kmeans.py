'''
KMeans module which selects random pixels from random frames and applies KMeans to the
average flow obtained and clusters the flows into 40 clusters. Call find_labels() method
to obtain the cluster number of each pixel of every frame as a list of dimensions,
Number of frames X 20 X 20
'''
import random
import numpy as np
from cv2 import VideoCapture
from sklearn.cluster import KMeans
from get_frames import get_random_frames
from get_flow import get_flow


def pick_random_pixels(flow):
    '''Selects random pixels from a given flow'''
    return random.sample(np.reshape(flow, (400, 2)), 40)


def pick_random_frames(frames):
    '''Selects random frames from the set of all frames (got by calling
    get_random_frames() method) and calls get_flow() method to get the
    average flow of a frame and then calls pick_random_pixels()'''
    data = []
    random_frames = random.sample(frames, len(frames)/10.0)
    for frame in random_frames:
        flow = np.array(get_flow(VideoCapture(frame[0]), frame[1]))
        pixels = pick_random_pixels(flow)
        for pixel in pixels:
            data.append(pixel)
    return data


def find_clusters(frames):
    '''Find the 40 clusters using K-Means method'''
    data = np.array(pick_random_frames(frames))
    kmeans = KMeans(40)
    kmeans.fit(data)
    return kmeans

def find_labels():
    '''Find the cluster number of every pixel of every frame'''
    frames = get_random_frames()
    kmeans = find_clusters(frames)
    labels = []
    for frame in frames:
        flow = np.array(get_flow(VideoCapture(frame[0]), frame[1]))
        labels.append(np.reshape(kmeans.predict(np.reshape(flow, (400, 2))), (20, 20)))
    return labels
