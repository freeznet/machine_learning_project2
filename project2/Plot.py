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

title = ['LR_Train','LR_Test','kNN_Train','kNN_Test','kNN+PCA_Train','kNN+PCA_Test']

y_pos = np.arange(6)


for data in datas:
    x_pos = []
    x_pos.append(LR_results[data]['p_train'])
    x_pos.append(LR_results[data]['p_test'])
    x_pos.append(kNN_results[data]['p_train'])
    x_pos.append(kNN_results[data]['p_test'])
    x_pos.append(kNN_results[data]['p_pca_train'])
    x_pos.append(kNN_results[data]['p_pca_test'])
    plt.figure(facecolor='#ffffff', figsize=(12, 6))
    for i in range(0,6):
        if i % 2 == 0:
            plt.barh(y_pos[i], x_pos[i], color='red', align='center', alpha=0.4, height=0.4)
        else:
            plt.barh(y_pos[i], x_pos[i], color='blue', align='center', alpha=0.4, height=0.4)
    plt.text(70, -1, r'$k='+str(kNN_results[data]['select_k'])+', PCA: '+str(kNN_results[data]['select_dim'])+'$',fontdict=font)
    plt.yticks(y_pos, title)
    plt.grid(True)
    plt.xlabel('Accuracy (%)', fontdict=font)
    plt.title('Accuracy for dataset ['+data+']', fontdict=font)


plt.show()