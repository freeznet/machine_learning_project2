__author__ = 'Rui'

#from scipy.optimize import fmin_bfgs, fmin_cg, fmin_ncg, fmin_l_bfgs_b, fmin
from scipy import optimize as op
import numpy as np

from dataloader import *
from pylab import *

def sigmoid(X):
    den = 1.0 + e ** (-1.0 * X)
    d = 1.0 / den
    return d

class LogisticReg:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.set_data(xTrain, yTrain, xTest, yTest)
        self.theta = np.zeros(self.x_train.shape[1])
        self.J = None
        self.grad = None

    def costFunction(self, theta):
        theta = reshape(theta, (len(theta),1))
        self.J = (1.0 / self.n) * (-self.y_train.T.dot(log(sigmoid(self.x_train.dot(theta)))) - (1.0 - self.y_train).T.dot(log(1.0 - sigmoid(self.x_train.dot(theta)))))
        self.grad = ((1.0 / self.n) * (sigmoid(self.x_train.dot(theta)).T - self.y_train).T.dot(self.x_train)).T
        return self.J[0][0]

    def predict(self, X, theta):
        return (sigmoid(X.dot(c_[theta]))>=0.5)

    def minimum(self):
        options = {'full_output':True, 'maxiter':500}
        theta, cost, _, _, _, =  op.fmin(lambda t: self.costFunction(t), self.theta, **options)
        return theta

    def minimum_(self):
        theta =  op.fmin_bfgs(self.costFunction, self.theta)
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

    for one in dataset.database:
        print one
        data = dataset.load(one,0)
        XTrain = data.features
        YTrain = data.labels

        data = dataset.load(one,1)
        XTest = data.features
        YTest = data.labels

        lr = LogisticReg(xTrain=XTrain, yTrain=YTrain, xTest=XTest, yTest=YTest)

        Jinit = lr.costFunction(lr.theta)

        theta = lr.minimum_()
        #print '\nCost at theta found by fmin: %g' % cost
        #print '\nParameters theta:', theta

        p_train = lr.training_reconstruction(theta)
        p_test = lr.test_predictions(theta)

        print '\nAccuracy on training set: %g' % p_train
        print 'Accuracy on test set: %g' % p_test

        #print theta



        #lr.winningrate()
        #break



    #print "Initial betas:"
    #print lr.betas
    #print "Initial likelihood:"
    #print lr.lik(lr.betas)

    #print "Final betas:"
    #print lr.betas
    #print "Final lik:"
    #print lr.lik(lr.betas)

    #subplot(1, 2, 0 + 1)
    #lr.plot_training_reconstruction()
    #subplot(1, 2, 0 + 2)
    #lr.plot_test_predictions()



    #show()



