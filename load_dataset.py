from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import os
import numpy as np
import pickle
import traceback
import random
from sklearn.cluster import KMeans

from read_flo import read_flo

kmeans = pickle.load( open('kmeans.data', 'rb') )

def list_data(directory):
    return [(os.path.join(root, f),
                os.path.join(root.replace('final', 'flow'), f.replace('.png', '.flo')))
            for root, _, files in os.walk(directory + '/final') for f in files
            if True]

train_files = [], val_files = []
def ldata(directory):
    global train_files, val_files
    if len(train_files) != 0:
        return
    for root, _, files in os.walk(directory + '/final'):
        count = 0
        for f in files:
            imgpath = os.path.join(root, f)
            flopath = os.path.join(root.replace('final', 'flow'), f.replace('.png', '.flo'))
            if count > len(files) * 0.8:
                val_files.append( (imgpath, flopath) )
            else:
                train_files.append( (imgpath, flopath) )


def transform_flow_to_out(flow):
    outflow = np.empty([20, 20], dtype=np.int32)
    for i in range(20):
        for j in range(20):
            flowvec = np.empty([2])
            flowvec[0] = np.average(flow[10*i:10*i+10, 10*j:10*j+10, 0])
            flowvec[1] = np.average(flow[10*i:10*i+10, 10*j:10*j+10, 1])
            outflow[i][j] = kmeans.predict(flowvec.reshape(1, -1))
    return np_utils.to_categorical(outflow.reshape(400), num_classes=40)

def random_crop((image, flow), crop_size):
    height, width = image.shape[1:]
    dy, dx = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return (image[:, y:(y+dy), x:(x+dx)], flow[x:(x+dx), y:(y+dy), :])

def image_generator(list_of_files, crop_size):
    while True:
        tup = random.choice(list_of_files)
        try:
            img = img_to_array(load_img(tup[0]))
            flow = read_flo(tup[1])
        except:
            traceback.print_exc()
            continue
            return
        # print(img.shape, flow.shape)
        (cropped_img, cropped_flow) = random_crop((img, flow), crop_size)
        if cropped_img is None:
            continue
        # print(cropped_img.shape, cropped_flow.shape)
        out_flow = transform_flow_to_out(cropped_flow)
        # print(cropped_img.shape, out_flow.shape)
        yield (cropped_img, out_flow)

def group_by_batch(dataset, batch_size):
    while True:
        print('group_by_batch')
        try:
            sources, targets = zip(*[next(dataset) for i in xrange(batch_size)])
            print(targets[0].shape, len(targets))
            batch = (np.stack(sources), np.stack(targets))
            # 32*400*onehotencoding
            print(batch[1].shape)
            yield batch
        except:
            traceback.print_exc()
            return

def load_dataset(directory, batch_size, wantVal=False):
    global train_files, val_files
    crop_size = (200, 200)
    ldata(directory)
    files = val_files if wantVal else train_files
    generator = image_generator(files, crop_size)
    if batch_size != 1:
        generator = group_by_batch(generator, batch_size)
    return generator

# for (image, flow) in load_dataset('datasets/Sintel/training'):
#     print(list_data('datasets/Sintel/training'))
