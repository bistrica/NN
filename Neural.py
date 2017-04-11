import numpy
import sklearn

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from theano.tensor.signal import pool
from wosedon.mergers.synsetslumerger2 import SynsetsLUMerger2
from wosedon.basegraph import BaseGraph
from PLWNGraphBuilder import PLWNGraph
import MySQLdb
from sklearn.neural_network import MLPClassifier
from summarizer import Finder



class NeuralNet(object):
    RELATIONS=24
    clf=None
    res=''


    def create_neural(self,attributes, labels, data, data_labels):

        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(7, 2), random_state=1)#5,2

        self.clf.fit(attributes, labels)
        ccc=0
        for i in range(len(data)):
        #    print self.clf.predict(data[i]), data_labels[i]
            if self.clf.predict(data[i])==data_labels[i]:
                ccc+=1
        print 'res: ',ccc,'/',len(data)
        self.res=self.res+', res: '+str(ccc)+'/'+str(len(data))
        #return result
        #print '>', clf.predict([[2., 2.], [-1., -2.]])

    def create_conv(self,attributes, labels, data, results):
        X_train=attributes
        y_train=labels
        X_test=data
        y_test=results
        net1 = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('conv2d1', layers.Conv2DLayer),
                    ('maxpool1', layers.MaxPool2DLayer),
                    ('conv2d2', layers.Conv2DLayer),
                    ('maxpool2', layers.MaxPool2DLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('dense', layers.DenseLayer),
                    ('dropout2', layers.DropoutLayer),
                    ('output', layers.DenseLayer),
                    ],
            # input layer
            input_shape=(None, 1, 1, self.RELATIONS*3),
            # layer conv2d1
            conv2d1_num_filters=32,
            conv2d1_filter_size=(5, 5),
            conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv2d1_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=(2, 2),
            # layer conv2d2
            conv2d2_num_filters=32,
            conv2d2_filter_size=(5, 5),
            conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
            # layer maxpool2
            maxpool2_pool_size=(2, 2),
            # dropout1
            dropout1_p=0.5,
            # dense
            dense_num_units=256,
            dense_nonlinearity=lasagne.nonlinearities.rectify,
            # dropout2
            dropout2_p=0.5,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=4,
            # optimization method params
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,
            max_epochs=10,
            verbose=1,
            )
        # Train the network
        nn = net1.fit(X_train, y_train)
        preds = net1.predict(X_test)

        cm = confusion_matrix(y_test, preds)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def predict(self,data):
        result=self.clf.predict(data)
        return result

    def compare(result, ground_truth):
        good=0
        bad=0
        for i in range(len(result)):
            if result[i]==ground_truth[i]:
                good+=1
            else:
                bad+=1
        return [good, bad]

    #set=load_dataset()
    #print 'so ',set[0]
    #print 's1 ',set[1]
    #print 'len ', len(set[0]), ' ',len(set[1])
    #result=createNeural(set[0], set[1],set[4])
    #result=create_conv(set[0], set[1], set[4], set[5])



    #print ': ',result#predict(set[4])
    #print ':: ',set[5]
    #print '>>> ',compare(result,set[5])

    #polaryzacja
    # [syno+, syno-, hipo+, hipo-, hiper+, hiper-, anto+, anto-]

    #dobry [ , , , , , ]  1
    #zly [ 0, 1, 0, 4, 2,0] 0
    #madry [ , , , , , ]
    #glupi [ , , , , , ]
    #glina [ , , , , , ]
    #pies [ , , , , , ]
    #policjant [ , , , , , ]


    #a=
    #createNeural(a,l)
    #predict(d)