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
import theano.tensor as T


class Neural(object):
    RELATIONS=24
    clf=None
    res=''
    graph=None
    propagator=None
    hidden_layers=(32,16,8)
    X_train=[]
    X_test = []
    Y_train = []
    Y_test = []

    def __init__(self,propagator,hidden_layers):
        self.graph=propagator.GRAPH
        self.propagator=propagator
        self.hidden_layers=tuple(hidden_layers)
        print 'hidde ',hidden_layers

    def create_mlp(self):
        mlp=self.build_mlp()
        print 'kel ',len(self.X_train),len(self.Y_train)
        mlp.fit(self.X_train,self.Y_train,200)
        preds = mlp.predict(self.X_test)
        cm = confusion_matrix(self.Y_test, preds)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def create_neural(self):#,attributes, labels, data, data_labels):
        print 'lenn ',len(self.X_train), len(self.Y_train), len(self.X_test), len(self.Y_test)
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=self.hidden_layers, random_state=1)#5,2

        self.clf.fit(self.X_train, self.Y_train)
        #self.predict_test_set()
        #return result
        #print '>', clf.predict([[2., 2.], [-1., -2.]])

    def predict_test_set(self):
        ccc = 0
        for i in range(len(self.X_test)):
            #    print self.clf.predict(data[i]), data_labels[i]
            #print 'y_t :',self.Y_train
            #print 'lenY ',len(self.Y_train)#TODOTypeError: object of type 'int' has no len()
            if self.clf.predict(self.X_test[i]) == self.Y_test[i]:
                ccc += 1
        print 'res: ', ccc, '/', len(self.Y_test)
        self.res = self.res + ', res: ' + str(ccc) + '/' + str(len(self.Y_test))

    def create_data(self,percent):
        X_train = list()
        Y_train = list()
        for pol in self.graph.list_of_polar.keys():
            vec, label = self.propagator.get_vector(self.graph.lu_nodes[pol])
            if vec is None:
                continue
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
        print '..', len(self.X_train), len(self.X_test), len(self.Y_train), len(self.Y_test)
        return self.X_train, self.X_test, self.Y_train, self.Y_test# X[0],X[1],Y[0],Y[1]

    def append_training_item(self,item,label):
        self.X_train.append(item)
        self.Y_train.append(label)


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



    def build_mlp(self,input_var=None):



        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)
        #self.X_train = self.X_train.reshape((len(self.X_train),1,len(self.X_train[0])))

        #self.X_test = self.X_test.reshape((len(self.X_test),1, len(self.X_test[0])))
        shape = self.X_train.shape#[1:]
        print 'shape ',shape,self.X_train.shape,self.X_train[0].shape
        sha=self.X_train.shape#[0].shape
        #input_var=np.asarray(self.X_train[0].shape)
        #Y = T.lvector()
        #input_var=T.tensor3('inputs')
        l_in = lasagne.layers.InputLayer(shape=sha,
                                         input_var=None)

        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

        l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

        l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

        l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

        net = NeuralNet(l_out, update_learning_rate=0.01)

        return net
