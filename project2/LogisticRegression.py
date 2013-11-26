from numpy.numarray import reshape
from numpy import c_

__author__ = 'Rui'

#from scipy.optimize import fmin_bfgs, fmin_cg, fmin_ncg, fmin_l_bfgs_b, fmin
import time, cPickle, sys
import numpy as np
from scipy import optimize as op


if sys.platform == 'win32':
    default_timer = time.clock
else:
    default_timer = time.time

np.seterr(all='ignore')

from dataloader import *

def sigmoid(X):
    den = 1.0 + np.exp(-1.0 * X)
    d = 1.0 / den
    return d

class LogisticReg:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.set_data(xTrain, yTrain, xTest, yTest)
        self.theta = np.zeros(self.x_train.shape[1])
        self.J = None

    def costFunction(self, theta):
        theta = reshape(theta, (len(theta),1))

        h = sigmoid(self.x_train.dot(theta))

        self.J = (1.0 / self.n) * (-self.y_train.T.dot(np.log(h)) - (1.0 - self.y_train).T.dot(np.log(1.0 - h)))

        return self.J[0][0]


    def predict(self, X, theta):
        return (sigmoid(X.dot(c_[theta]))>=0.5)

    def minimum(self):
        options = {'full_output':True, 'maxiter':500}
        theta, cost, _, _, _, =  op.fmin(lambda t: self.costFunction(t), self.theta, **options)
        return theta

    def minimum_auto(self):
        theta =  op.fmin_bfgs(self.costFunction, self.theta, disp=False)
        return theta

    def set_data(self, x_train, y_train, x_test, y_test):
        self.x_train = c_[np.ones(x_train.shape[0]), x_train] # add intercept terms
        self.y_train = y_train
        self.x_test = c_[np.ones(x_test.shape[0]), x_test]
        self.y_test = y_test
        self.n = y_train.shape[0] # num of samples / training
        self.testn = y_test.shape[0] # num of samples / testing

    def training_reconstruction(self, theta):
        p = self.predict(self.x_train, theta)
        return (p == self.y_train).mean()*100.0

    def test_predictions(self, theta):
        p = self.predict(self.x_test, theta)
        return (p == self.y_test).mean()*100.0

if __name__ == "__main__":
    dataset = Dataset()
    results = {}

    for one in dataset.database:
        print 'Current dataset:',one
        result = {}
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
        lr = LogisticReg(xTrain=XTrain, yTrain=YTrain, xTest=XTest, yTest=YTest)

        Jinit = lr.costFunction(lr.theta)

        theta = lr.minimum_auto()

        p_train = lr.training_reconstruction(theta)
        p_test = lr.test_predictions(theta)
        result['t_preform_train'] = default_timer() - currentTime
        print '     Data training done. (%0.3fs)'%(default_timer() - currentTime)

        print '     Accuracy on training set: %g' % p_train
        print '     Accuracy on test set: %g' % p_test
        result['p_train'] = p_train
        result['p_test'] = p_test
        result['t_overall'] = default_timer() - initRuntime
        print '     Total runtime: %0.3fs'%(default_timer() - initRuntime)
        print
        results[one] = result

    print 'Dumping result data...'
    f = file('LogisticRegression.sav', 'wb')
    parameters = (results)
    cPickle.dump(parameters, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print 'done.'



