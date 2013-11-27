__author__ = 'Rui'

import numpy as np
import cPickle
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

## draw LogisticRegression

f = file('LogisticRegression.sav', 'rb')
(LR_results) = cPickle.load(f)
f.close()

#print LR_results


## draw kNearestNeighbor

f = file('kNearestNeighbor.sav', 'rb')
(kNN_results) = cPickle.load(f)
f.close()

#print kNN_results

datas = [x for x in kNN_results]

print datas

title = ['LR_Train','LR_Test','kNN_Train','kNN_Test','kNN+PCA_Train','kNN+PCA_Test','SVM_Linear_Train','SVM_Linear_Test','SVM_RBF_Train','SVM_RBF_Test']

y_pos = np.arange(10)

# SVM data
SVM_results = {}

#C=0.0833
#vm_learn -t 2 -g 0.70 ionosphere_train.dat i_model_gamma
#C=0.5152
IonosphereSet = {'p_train':91.6, 'p_test':87.13, 'select_gamma':0.7, 'p_gamma_train': 98.0, 'p_gamma_test':91.09}
SVM_results['Ionosphere'] = IonosphereSet


#C=0.0039
#svm_learn -t 2 -g 0.01 isolet_train.dat
#C=0.5431
ISOLETSet = {'p_train':100.0, 'p_test':98.33, 'select_gamma':0.01, 'p_gamma_train': 100.0, 'p_gamma_test':98.33}
SVM_results['ISOLET'] = ISOLETSet

#C=0.4500
#svm_learn -t 2 -g 300 liver_train.dat lmodel
#C=0.5000
LiverSet = {'p_train':58.5, 'p_test':57.24, 'select_gamma':300.0, 'p_gamma_train': 59.0, 'p_gamma_test':57.93}
SVM_results['Liver'] = LiverSet

#C=0.0000
#svm_learn -t 2 -g 0.0000005 mnist_train.dat model
#C=0.5362

MNISTSet = {'p_train':98.75, 'p_test':96.0, 'select_gamma':0.0000005, 'p_gamma_train': 100.0, 'p_gamma_test':98.25}
SVM_results['MNIST'] = MNISTSet

#C=0.0476 0.22s
#svm_learn -t 2 -g 0.25 mushroom_train.dat model
#C=0.5026 4.75s
MushroomSet = {'p_train':99.98, 'p_test':99.83, 'select_gamma':0.25, 'p_gamma_train': 100.0, 'p_gamma_test':99.93}
SVM_results['Mushroom'] = MushroomSet


for data in datas:
    x_pos = []
    x_pos.append(LR_results[data]['p_train'])
    x_pos.append(LR_results[data]['p_test'])
    x_pos.append(kNN_results[data]['p_train'])
    x_pos.append(kNN_results[data]['p_test'])
    x_pos.append(kNN_results[data]['p_pca_train'])
    x_pos.append(kNN_results[data]['p_pca_test'])
    x_pos.append(SVM_results[data]['p_train'])
    x_pos.append(SVM_results[data]['p_test'])
    x_pos.append(SVM_results[data]['p_gamma_train'])
    x_pos.append(SVM_results[data]['p_gamma_test'])

    plt.figure(facecolor='#ffffff', figsize=(12, 6))
    for i in range(0,10):
        if i % 2 == 0:
            plt.barh(y_pos[i], x_pos[i], color='red', align='center', alpha=0.4, height=0.4)
        else:
            plt.barh(y_pos[i], x_pos[i], color='blue', align='center', alpha=0.4, height=0.4)
    plt.text(60, -1, r'$k='+str(kNN_results[data]['select_k'])+', PCA: '+str(kNN_results[data]['select_dim'])+', Gamma='+str(SVM_results[data]['select_gamma'])+'$',fontdict=font)
    plt.yticks(y_pos, title)
    plt.grid(True)
    plt.xlabel('Accuracy (%)', fontdict=font)
    plt.title('Accuracy for dataset ['+data+']', fontdict=font)
    print data
    print x_pos
    print kNN_results[data]['select_k']
    print '------------------------'
    plt.savefig(data+'.png')


plt.show()