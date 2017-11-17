'''
KMeans module which selects random pixels from random frames and applies KMeans to the
average flow obtained and clusters the flows into 40 clusters. Call find_labels() method
to obtain the cluster number of each pixel of every frame as a list of dimensions,
Number of frames X 20 X 20
'''
import random
import os
# import cv2
import numpy as np
from sklearn.cluster import KMeans
from read_flo import read_flo
# from get_flow import get_flow
import pickle


def pick_random_pixel_flow(flows):
    '''Selects random pixels from a given flow'''
    # return random.sample(np.reshape(flow, (400, 2)), 40)
    data = np.array([0.0, 0.0])
    for flow in flows:
        height = len(flow)
        width = len(flow[0])
        data = np.vstack((data, random.sample(np.reshape(flow, (width*height, 2)), width*height//10)))
    return data[1:]


# def pick_random_frames(frames):
#     '''Selects random frames from the set of all frames (got by calling
#     get_random_frames() method) and calls get_flow() method to get the
#     average flow of a frame and then calls pick_random_pixels()'''
#     data = []
#     random_frames = random.sample(frames, len(frames) // 10)
#     for frame in random_frames:
#         frame_numbers = random.sample(frame[1], len(frame[1]) // 10)
#         for frame_number in frame_numbers:
#             flow = np.array(get_flow(cv2.VideoCapture(frame[0]), frame_number))
#             pixels = pick_random_pixels(flow)
#             for pixel in pixels:
#                 data.append(pixel)
#     return data


def find_clusters(flows):
    '''Find the 40 clusters using K-Means method'''
    # data = np.array(pick_random_frames(frames))
    data = np.array(pick_random_pixel_flow(flows))
    kmeans = KMeans(40)
    kmeans.fit(data)
    return kmeans

# def process_frame(image_path):
#     '''Returns a numpy array of the image. Note: Size is the same as the default image.
#     Not resized'''
#     image = cv2.imread(image_path)
#     return image

def get_frames_and_flow(basepath):
    '''Get the frames and flow from given path'''
    # final_path = os.path.join(basepath, 'final')
    flows_path = os.path.join(basepath, 'flow')
    action_names = os.listdir(flows_path)
    # images = []
    flows = []
    for action in action_names:
        action_path = os.path.join(flows_path, action)
        flows_list = os.listdir(action_path)
        random_flows_list = random.sample(flows_list, len(flows_list) // 10)
        for random_flow in random_flows_list:
            random_flow_path = os.path.join(action_path, random_flow)
            # random_image_path = os.path.join(final_path, action, random_flow)
            # random_image_path = random_image_path[:-3] + 'png'
            # images.append(cv2.imread(random_flow_path))
            flows.append(read_flo(random_flow_path))
    # return images, flows
    return flows



def find_labels(datapath):
    '''Find the cluster number of every pixel of every frame'''
    # images, flows = get_frames_and_flow('./datasets/UCF-processed')
    random_flows = get_frames_and_flow(datapath)
    kmeans = find_clusters(random_flows)
    labels = []
    flows_path = os.path.join(datapath, 'flow')
    for action in os.listdir(flows_path):
        flows_list = os.listdir(os.path.join(flows_path, action))
        label = []
        for flow_name in flows_list:
            flow = read_flo(os.path.join(flows_path, action, flow_name))
            height = len(flow)
            width = len(flow[0])
            label.append(np.reshape(kmeans.predict(np.reshape(flow, (width*height, 2))),
                                    (height, width)))
        labels.append(label)
    # for frame in frames:
        # flow = np.array(get_flow(VideoCapture(frame[0]), frame[1]))
        # labels.append(np.reshape(kmeans.predict(np.reshape(flow, (400, 2))), (20, 20)))
    return labels, kmeans

labels, kmeans = find_labels('datasets/UCF-processed/training')
pickle.dump(labels, open('labels.data', 'wb'))
pickle.dump(kmeans, open('kmeans.data', 'wb'))
