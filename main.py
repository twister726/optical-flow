from get_frames import get_random_frames
from kmeans import find_labels
import pickle

framelist = get_random_frames()
trainsize = len(framelist) * 0.8
labels, kmeans = find_labels(framelist)
# Train using framelist[:trainsize] and labels[:trainsize]
# Test using framelist[trainsize+1:] and labels[trainsize+1:]
pickle.dump(framelist, open( "framelist.dump", "wb" ) )
pickle.dump(labels, open( "labels.dump", "wb" ) )
pickle.dump(kmeans, open( "kmeans.dump", "wb" ) )
