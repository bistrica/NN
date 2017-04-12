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
    graph=None
    propagator=None
    X_train=[]
    X_test = []
    Y_train = []
    Y_test = []

    def __init__(self,graph,propagator):
        self.graph=graph
        self.propagator=propagator


    def create_neural(self):#,attributes, labels, data, data_labels):
        print 'lenn ',len(self.X_train), len(self.Y_train)
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(7, 2), random_state=1)#5,2

        self.clf.fit(self.X_train, self.Y_train)

        #return result
        #print '>', clf.predict([[2., 2.], [-1., -2.]])

    def predict_test_set(self):
        ccc = 0
        for i in range(len(self.Y_train)):
            #    print self.clf.predict(data[i]), data_labels[i]
            if self.clf.predict(self.Y_train[i]) == self.Y_test[i]:
                ccc += 1
        print 'res: ', ccc, '/', len(self.Y_train)
        self.res = self.res + ', res: ' + str(ccc) + '/' + str(len(self.Y_train))

    def create_data(self,percent):
        X_train = list()
        Y_train = list()
        for pol in self.graph.list_of_polar.keys():
            vec, label = self.propagator.get_vector(self.graph.lu_nodes[pol])
            vec = numpy.asarray(vec)
            # vec=vec.reshape(-1, 1)
            # print 'VC ',vec.ndim
            X_train.append(vec)
            Y_train.append(label)
        X = [X_train[:int(percent * len(X_train))], X_train[int(percent * len(X_train)):]]
        Y = [Y_train[:int(percent * len(Y_train))], Y_train[int(percent * len(Y_train)):]]
        print '.',len(X[0]),len(X[1])
        print len(Y[0]), len(Y[1])
        self.X_train=X[0]
        self.Y_train=Y[0]
        self.X_test=X[1]
        self.Y_test=Y[1]
        return self.X_train, self.X_test, self.Y_train, self.Y_test# X[0],X[1],Y[0],Y[1]

    def append_training_item(self,item):
        self.X_train=item

    def append_training_label(self,item):
        self.X_test=item

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
        if isinstance(data,list):
            result=[]
            for item in data:
                result.append(self.clf.predict(data))
        else:
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