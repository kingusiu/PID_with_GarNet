import pandas as pd
import numpy as np

features = ['event', 'trackster', 'x', 'y', 'z', 'r', 'layer', 'E', 'nHits', 'phi', 'eta' ]

def read_dataset_by_event( filepath ):

    data = pd.read_hdf( filepath )
    #data = data.head(1000)
    data = data.loc[ data['trackster'] != 0.0 ] # filter out all dangling layer clusters
    data = data[ features ] # select relevant features
    data.set_index('event',inplace=True)
    events = data.groupby('event')
    return events.apply(pd.Series.tolist).tolist()  # unpadded per-event dataset (list of lists)


def find_max_number_of_2d_layer_clusters( datasets ):

    most_layerclusters = max( datasets, key = lambda data : len(max(data,key=len)))
    return len(max(most_layerclusters,key=len))


def get_number_of_features( ):

    return len(features) - 1 # 'events' feature dropped


def get_padded_dataset( dataset, max_num_2dcluster, num_features ):

    x = np.array([])

    for event in dataset:
        next = np.zeros((max_num_2dcluster, num_features))  # create dummy 2d slice of zeros
        next[:len(event)] = event # fill all valid 2d layer clusters along axis 0
        x = np.vstack((x, next[np.newaxis, ...])) if x.size else next[np.newaxis, ...] # stack padded events

    return x

