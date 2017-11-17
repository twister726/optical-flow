#!/usr/bin/env python3
import os
import pickle
# import zlib
import numpy as np

def list_files(directory):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.endswith('.flow')]

def transform_proto(directory):
    for f in list_files(directory):
        with open(f, 'rb') as fr:
            flow = pickle.load(fr)
        os.remove(f)
        np.savez_compressed(f, flow)
        print(f)

transform_proto('datasets/UCF-processed/training/flow')
transform_proto('datasets/UCF-processed/test/flow')
