import numpy as np
import os
import pickle

def read_flo(fpath):
    if os.path.exists(fpath + 'w'): # Numpy pickle dump for ucf preprocessed
        return pickle.load( open(fpath + 'w', 'rb') )
    with open(fpath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (w, h, 2))
            return data2D
            # data3D = np.dstack( ( data2D, np.zeros((w, h)) ) )
            # return data3D
