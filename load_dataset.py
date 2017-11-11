from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

from read_flo import read_flo

def list_data(directory):
    return [(os.path.join(root, f),
                os.path.join(root.replace('final', 'flow'), f.replace('.png', '.flo')))
            for root, _, files in os.walk(directory + '/final') for f in files
            if True]

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
        yield (cropped_img, cropped_flow)

def group_by_batch(dataset, batch_size):
    while True:
        try:
            sources, targets = zip(*[next(dataset) for i in xrange(batch_size)])
            batch = (np.stack(sources), np.stack(targets))
            yield batch
        except:
            return

def load_dataset(directory, crop_size, batch_size):
    files = list_data(directory)
    generator = image_generator(files, crop_size)
    if batch_size != 1:
        generator = group_by_batch(generator, batch_size)
    return generator

for (image, flow) in load_dataset('datasets/Sintel/training'):
    print(list_data('datasets/Sintel/training'))
