# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.stats
import scipy.io
import urllib
import logging
import os

class DataLoader:
    def __init__(self, filename, norm=0, needlist = None):
        mat = scipy.io.loadmat('../datasets/' + filename)

        # do normalize if needed for feature data
        if norm>0:
            mat['X'] = mat['X']/1.0

        if not needlist == None:
            self.features = mat['X'][needlist,:]
            self.labels = mat['Y'][needlist,:]
        else:
            self.features = mat['X']
            self.labels = mat['Y']

        if not type(self.features) == np.ndarray:
            self.features = self.features.toarray()

class DataSetter:
    def __init__(self, feature, label):
        self.features = feature
        self.labels = label

class Dataset:
    def __init__(self):
        self.database = {'Ionosphere':['ionosphere_test.mat','ionosphere_train.mat'], 'ISOLET':['isolet_test.mat', 'isolet_train.mat'], 'Liver':['liver_test.mat', 'liver_train.mat'], 'MNIST':['mnist_test.mat', 'mnist_train.mat'], 'Mushroom':['mushroom_test.mat', 'mushroom_train.mat']}
        for dataset in self.database:
            for f in self.database[dataset]:
                print('Checking %s database file %s' % (dataset, f))
                target = '../datasets/'+f
                origin = 'https://github.com/freeznet/machine_learning_project2/blob/master/datasets/' + f + '?raw=true'
                if (not os.path.isfile(target)):
                    print('Downloading %s %s data from online site.' % (dataset, f ))
                    urllib.urlretrieve(origin, target)

    def load(self, data, type):
        return DataLoader(self.database[data][type])