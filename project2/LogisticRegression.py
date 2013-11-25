__author__ = 'Rui'

from scipy.optimize import fmin_bfgs, fmin_cg, fmin_ncg, fmin_l_bfgs_b
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
        self.betas = np.zeros(self.x_train.shape[1])
        print type(xTrain), type(self.betas)

    def negative_lik(self, betas):
        return -1 * self.lik(betas)

    def lik(self, betas):
        """ Likelihood of the data under the current settings of parameters. """
        l = 0
        for i in range(self.n):
            l += log(sigmoid(self.y_train[i] * np.dot(betas, self.x_train[i,:])))
            #l += self.y_train[i] * log(sigmoid(np.dot(betas, self.x_train[i,:]))) + (1.0 - self.y_train[i]) * log(1.0 - sigmoid(np.dot(betas, self.x_train[i,:])))
        return l

    def train(self):
        dB_k = lambda B, k : - np.sum([ \
            self.y_train[i] * self.x_train[i, k] * \
            sigmoid(-self.y_train[i] *\
            np.dot( B, self.x_train[i,:])) \
            for i in range(self.n)])

        dB = lambda B : np.array([dB_k(B, k) \
            for k in range(self.x_train.shape[1])])

        self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)

    def set_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n = y_train.shape[0]
        self.testn = y_test.shape[0]

    def training_reconstruction(self):
        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i,:])) >= 0.5
        return p_y1

    def test_predictions(self):
        p_y1 = np.zeros(self.testn)
        for i in range(self.testn):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i,:])) >= 0.5
        return p_y1

    def plot_training_reconstruction(self):
        plot(np.arange(self.n), self.y_train, 'bo')
        plot(np.arange(self.n), self.training_reconstruction(), 'rx')
        ylim([-.1, 1.1])

    def plot_test_predictions(self):
        plot(np.arange(self.testn), self.y_test, 'yo')
        plot(np.arange(self.testn), self.test_predictions(), 'rx')
        ylim([-.1, 1.1])

    def winningrate(self):
        training_ret = self.training_reconstruction()
        test_ret = self.test_predictions()
        print (training_ret==self.y_train).mean() * 100.0
        print (test_ret==self.y_test).mean() * 100.0



if __name__ == "__main__":
    dataset = Dataset()

    for one in dataset.database:
        data = dataset.load(one,0)
        XTrain = data.features
        YTrain = data.labels

        data = dataset.load(one,1)
        XTest = data.features
        YTest = data.labels

        lr = LogisticReg(xTrain=XTrain, yTrain=YTrain, xTest=XTest, yTest=YTest)

        lr.train()

        print one
        lr.winningrate()



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



