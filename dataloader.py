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
        mat = scipy.io.loadmat(filename)

        # do normalize if needed for feature data
        if norm>0:
            mat['X'] = mat['X']/1.0

        if not needlist == None:
            self.features = mat['X'][needlist,:]
            self.labels = mat['Y'][needlist,:]
        else:
            self.features = mat['X']
            self.labels = mat['Y']

class DataSetter:
    def __init__(self, feature, label):
        self.features = feature
        self.labels = label

class Dataset:
    def __init__(self):
        self.database = {'Ionosphere':['ionosphere_test.mat','ionosphere_train.mat'], 'ISOLET':['isolet_test.mat', 'isolet_train.mat'], 'Liver':['liver_test.mat', 'liver_train.mat'], 'MNIST':['mnist_test.mat', 'mnist_train.mat'], 'Mushroom':['mushroom_test.mat', 'mushroom_train.mat']}
        for dataset in self.database:
            for f in self.database[dataset]:
                logging.info('Checking %s database file %s' % (dataset, f))
                origin = 'http://all.aboutfree.me/'
                if (not os.path.isfile(origin)):
                    logging.info('Downloading %s %s data from online site.' % (dataset, f ))
                    urllib.urlretrieve(origin, file)
class DataSet:
    def __init__(self):
        self.database = ['MNIST', 'MNIST', 'COIL20']
        self.datafiles = ['10kTrain.mat', 'Test.mat', 'COIL20.mat']
        self.dataorigin = ['http://www.cad.zju.edu.cn/home/dengcai/Data/MNIST/10kTrain.mat', 'http://www.cad.zju.edu.cn/home/dengcai/Data/MNIST/Test.mat', 'http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL20.mat']
        for file in self.datafiles:
            print 'Checking %s database file %s' % (self.database[self.datafiles.index(file)],file)
            if (not os.path.isfile(file)):
                origin = self.dataorigin[self.datafiles.index(file)]
                print 'Downloading %s %s data from %s' % (self.database[self.datafiles.index(file)],file,origin)
                urllib.urlretrieve(origin, file)
        self.MNISTTrain = DataLoader(self.datafiles[0], 255.0)
        self.MNISTTest = DataLoader(self.datafiles[1], 255.0)
        totallist = [x for x in range(0,1440)]
        trainneedlist = [6*x for x in range(0,240)]
        testneedlist = [x for x in totallist if x not in trainneedlist]

        self.COIL20Train = DataLoader(self.datafiles[2], 0, trainneedlist)
        self.COIL20Test = DataLoader(self.datafiles[2], 0, testneedlist)

        self.MNIST = {}
        self.MNIST[0] = self.MNISTTrain
        self.MNIST[1] = self.MNISTTest
        self.MNIST[2] = 255.0

        self.COIL20 = {}
        self.COIL20[0] = self.COIL20Train
        self.COIL20[1] = self.COIL20Test
        self.COIL20[2] = 1.0

        print 'Database file all done...'


