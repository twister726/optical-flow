from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans

from read_flo import read_flo

kmeans = pickle.load( open('kmeans.data', 'rb') )

def list_data(directory):
    return [(os.path.join(root, f),
                os.path.join(root.replace('final', 'flow'), f.replace('.png', '.flo')))
            for root, _, files in os.walk(directory + '/final') for f in files
            if True]

def transform_flow_to_out(flow):
    outflow = np.empty([20, 20], dtype=np.int32)
    for i in range(20):
        for j in range(20):
            flowvec = np.empty([2])
            flowvec[0] = np.average(flow[10*i:10*i+10, 10*j:10*j+10, 0])
            flowvec[1] = np.average(flow[10*i:10*i+10, 10*j:10*j+10, 1])
            outflow[i][j] = kmeans.predict(flowvec.reshape(1, -1))
    return outflow.reshape(400)

def random_crop((image, flow), crop_size):
    height, width = image.shape[1:]
    dy, dx = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return (image[:, y:(y+dy), x:(x+dx)], flow[:, y:(y+dy), x:(x+dx)])

def image_generator(list_of_files, crop_size):
    while True:
        tup = np.random.choice(list_of_files)
        try:
            img = img_to_array(load_img(tup[0]))
            flow = read_flo(tup[1])
        except:
            return
        (cropped_img, cropped_flow) = random_crop((img, flow), crop_size)
        if cropped_img is None:
            continue
        out_flow = transform_flow_to_out(cropped_flow)
        yield (cropped_img, out_flow)

def group_by_batch(dataset, batch_size):
    while True:
        try:
            sources, targets = zip(*[next(dataset) for i in xrange(batch_size)])
            batch = (np.stack(sources), np.stack(targets))
            yield batch
        except:
            return

def load_dataset(directory, batch_size):
    crop_size = (200, 200)
    files = list_data(directory)
    generator = image_generator(files, crop_size)
    if batch_size != 1:
        generator = group_by_batch(generator, batch_size)
    return generator

# for (image, flow) in load_dataset('datasets/Sintel/training'):
#     print(list_data('datasets/Sintel/training'))
