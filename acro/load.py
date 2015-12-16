import sys
sys.path.append('..')

import os
import numpy as np
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from lib.config import data_dir
import random

def streams(ntrain=None, val=0., batch_size=128, path=None,
            shuffle=True, filter_label=None):
    """

    :param ntrain: # of train examples to generate (including val) (None = use all training examples)
    :param val: percentage of train to use as validation examples (0. = no validation data)
    :param batch_size: size of a batch
    :param path: path to hdf5 data file.
    :param shuffle: to shuffle the training data
    :param filter_label: take for training only examples that have the mask
        set and the label equal to filter_label (None = don't filter)
    :return:
        tr_data training data
        tr_stream stream of the training data that shoud be used for batch training
        val_stream stream of the training data thta should be used for batch validation
    """
    val_stream = None
    if path is None:
        path = os.path.join(data_dir, 'jpg.hdf5')
    tr_data = H5PYDataset(path, which_sets=('train',))

    if ntrain is None:
        ntrain = tr_data.num_examples
    else:
        ntrain = min(ntrain, tr_data.num_examples)

    if filter_label is not None:
        tr_handle = tr_data.open()
        _,labels = tr_data.get_data(tr_handle, slice(0, tr_data.num_examples))
        y = labels[:,0]
        mask = labels[:,1]
        print '# mask',mask.sum()
        idxs = np.where(mask & (y == filter_label))[0]
        ntrain = min(ntrain, len(idxs))
    else:
        idxs = range(ntrain)

    if val <= 0.:
        if shuffle:
            tr_scheme = ShuffledScheme(examples=idxs[:ntrain], batch_size=batch_size)
        else:
            tr_scheme = SequentialScheme(examples=idxs[:ntrain], batch_size=batch_size)
        tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)
        nval = 0
    else:
        nval = int(ntrain*val)
        train_examples = random.sample(idxs, ntrain)
        val_examples = train_examples[(ntrain-nval):ntrain]
        train_examples = train_examples[:(ntrain-nval)]
        if shuffle:
            tr_scheme = ShuffledScheme(examples=train_examples, batch_size=batch_size)
        else:
            tr_scheme = SequentialScheme(examples=train_examples, batch_size=batch_size)
        tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)
        val_scheme = SequentialScheme(examples=val_examples, batch_size=batch_size)
        val_stream = DataStream(tr_data, iteration_scheme=val_scheme)

    return tr_data, tr_stream, val_stream, ntrain-nval, nval