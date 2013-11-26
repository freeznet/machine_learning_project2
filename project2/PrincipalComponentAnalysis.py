__author__ = 'Rui'

import numpy as np
import scipy
import scipy.stats
import scipy.io
import cPickle
import math
import time

#ignore some useless warnings...
import warnings
np.seterr(all='ignore')
warnings.simplefilter("ignore", np.ComplexWarning)

class PCA:
    def __init__(self, xData):
        self.features = xData
        self.n, self.d = self.features.shape
        self.covariance = np.cov(self.features,rowvar=0)
        self.eigenValue, self.eigenVector = np.linalg.eig(self.covariance)

        # sort the principal components in decreasing order of corresponding eigenvalue
        self.sorter = list(reversed(self.eigenValue.argsort(0)))

        self.eigenValue = self.eigenValue[self.sorter]

        self.eigenVector = self.eigenVector[:, self.sorter]

        dimreduction = 2 # start with 2 dim, 1 dim cannot get variance -_-
        varianceRate = 0.0
        while dimreduction<self.eigenVector.shape[0]:
            self.currentVector = self.eigenVector[:,:dimreduction]
            self.currentFeature = self.features.dot(self.currentVector)
            self.currentCovariance = np.cov(self.currentFeature,rowvar=0)
            self.currentEigenValue, _ = np.linalg.eig(self.currentCovariance)
            varianceRate = self.currentEigenValue.sum() / self.eigenValue.sum()
            if varianceRate>=0.9:    #shut when variance rate greater than 90%
                break
            dimreduction += 1

        self.dim = dimreduction

    def DimReduce(self, sampleData, dim):
        self.features = sampleData
        self.n, self.d = self.features.shape
        self.covariance = np.cov(self.features,rowvar=0)
        self.eigenValue, self.eigenVector = np.linalg.eig(self.covariance)
        self.sorter = list(reversed(self.eigenValue.argsort(0)))
        self.eigenValue = self.eigenValue[self.sorter]
        self.eigenVector = self.eigenVector[:, self.sorter]
        self.currentVector = self.eigenVector[:,:dim]
        self.currentFeature = self.features.dot(self.currentVector)
        return self.currentFeature

