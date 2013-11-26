__author__ = 'Rui'

import sys, operator
from PrincipalComponentAnalysis import *
if sys.platform == 'win32':
    default_timer = time.clock
else:
    default_timer = time.time

import random

np.seterr(all='ignore')
from dataloader import *

class KNN:
    def __init__(self, xTrain, yTrain, xTest, yTest, pca=False):
        self.set_data(xTrain, yTrain, xTest, yTest)
        self.PCApreform = pca

    def set_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n = y_train.shape[0] # num of samples / training
        self.testn = y_test.shape[0] # num of samples / testing

    def predict(self, X, x, y, k):
        m = x.shape[0] # num of samples
        n = x.shape[1] # num of features

        prediction = np.zeros((X.shape[0], 1)) #

        for i in range(0, X.shape[0]):
            test_sample_matrix = (np.reshape(X[i,:], (X.shape[1], 1)).dot(np.ones((1, m)))).T

            distances = np.sum((x - test_sample_matrix)**2, axis=1)

            distances = distances ** 0.5 # sqrt

            mins = np.argsort(distances, axis=0)[:k] # min k distance

            minLabels =  y[mins] # min k labels

            uniqueLables = np.unique(minLabels) # class

            max = 0
            predicted_label = None
            for label in uniqueLables:
                crt_vote = np.sum(minLabels == label)
                if crt_vote>max:
                    max = crt_vote
                    predicted_label = label

            prediction[i] = predicted_label

        return prediction

    def training_reconstruction(self, predict):
        return (predict == self.y_train).mean()*100.0

    def test_predictions(self, predict):
        return (predict == self.y_test).mean()*100.0

    def analysis(self, predict, labels):
        return (predict == labels).mean()*100.0

def getRandomData(xdata, ydata, train_size):
    xlen = xdata.shape[0]
    ind = [x for x in range(0, xlen)] # generate all index of target dataset
    random.shuffle(ind) # shuffle the index list, let it randomly

    train_ind = ind[:int(xlen*train_size)]
    test_ind = ind[int(xlen*train_size):xlen]

    train_x = xdata[train_ind]
    train_y = ydata[train_ind]

    test_x = xdata[test_ind]
    test_y = ydata[test_ind]

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    dataset = Dataset()
    results = {}
    krange = [x for x in range(1,11)] # 1 .. 10
    for one in dataset.database:
        kset = {} # store cross-validation result to choose best K
        result = {}
        print 'Current dataset:',one
        initRuntime = default_timer()

        currentTime = default_timer()
        data = dataset.load(one,0)
        print '     Load Testing data done. (%0.3fs)'%(default_timer() - currentTime)
        result['t_load_test'] = default_timer() - currentTime
        XTrain = data.features
        YTrain = data.labels

        currentTime = default_timer()
        data = dataset.load(one,1)
        print '     Load Training data done. (%0.3fs)'%(default_timer() - currentTime)
        result['t_load_train'] = default_timer() - currentTime
        XTest = data.features
        YTest = data.labels

        currentTime = default_timer()

        print '     Create KNN classifier...'
        knn = KNN(xTrain=XTrain, yTrain=YTrain, xTest=XTest, yTest=YTest)

        print '     Start training ...'
        currentTime = default_timer()
        for k in krange:
            for i in range(0,10):   # preform 10 times cross-validation
                randomTrainX, randomTrainY, randomTestX, randomTestY = getRandomData(XTrain, YTrain, 0.8)
                trainPredict = knn.predict(randomTestX, randomTrainX, randomTrainY, k) # predict for test data with training data model
                trainResult = knn.analysis(trainPredict, randomTestY)
                kset[k] = kset.get(k, 0.0) + trainResult
            kset[k] = kset.get(k, 0.0) / 10.0 # -> get even value
            if kset.get(k, 0.0) >= 100.0:   # reach maximum, no need anymore testing
                break
        selectK = max(kset.iteritems(), key=operator.itemgetter(1))[0]
        print '     Get best k: %d with accuracy %f. (%0.3fs)'%(selectK, kset[selectK], default_timer() - currentTime)
        result['select_k'] = selectK

        currentTime = default_timer()
        testPredict = knn.predict(XTest, XTrain, YTrain, selectK)
        TrainPredict = knn.predict(XTrain, XTrain, YTrain, selectK)

        p_train = knn.training_reconstruction(TrainPredict)
        p_test = knn.test_predictions(testPredict)
        print '     %d-NN done. (%0.3fs)'%(selectK, default_timer() - currentTime)
        print '     [*] Accuracy on training set: %g' % p_train
        print '     [*] Accuracy on test set: %g' % p_test
        result['p_train'] = p_train
        result['p_test'] = p_test

        ### PCA
        currentTime = default_timer()
        print '     Estimating best parameter for PCA...'
        pca = PCA(XTrain) # using training data set
        bestPCA = pca.dim
        result['select_dim'] = bestPCA
        print '     Dim reduce from %d to %d.(%0.3fs)'%(XTrain.shape[1], bestPCA, default_timer() - currentTime)
        XDimReducedTrain = pca.currentFeature
        XDimReducedTest = pca.DimReduce(XTest, bestPCA)
        currentTime = default_timer()

        testPCAPredict = knn.predict(XDimReducedTest, XDimReducedTrain, YTrain, selectK)
        TrainPCAPredict = knn.predict(XDimReducedTrain, XDimReducedTrain, YTrain, selectK)

        p_train = knn.training_reconstruction(TrainPCAPredict)
        p_test = knn.test_predictions(testPCAPredict)
        result['p_pca_train'] = p_train
        result['p_pca_test'] = p_test
        print '     %d-NN with PCA (dim reduce to %d) done. (%0.3fs)'%(selectK, bestPCA, default_timer() - currentTime)
        print '     [*] Accuracy on training set: %g' % p_train
        print '     [*] Accuracy on test set: %g' % p_test
        result['t_overall'] = default_timer() - initRuntime
        print '     Total runtime: %0.3fs'%(default_timer() - initRuntime)
        print
        results[one] = result

    print 'Dumping result data...'
    f = file('kNearestNeighbor.sav', 'wb')
    parameters = (results)
    cPickle.dump(parameters, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print 'done.'







