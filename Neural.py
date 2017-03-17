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


db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="toor",  # your password
                     db="wordTEST")        # name of the data base


cur = db.cursor()

cur.execute("SELECT l.ID from lexicalunit l join lexicalunit l2 on l.lemma=l2.lemma where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %') and  (l2.comment like '%+ m' or l2.comment like '%+ s' or l2.comment like '%+ m %' or l2.comment like '%+ s %')")

not_disamb_list=list()

for row in cur.fetchall():
    not_disamb_list.append(row[0])

cur.execute("SELECT l.ID from lexicalunit l where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %')")

negative_list=list()#cur.fetchall()
for row in cur.fetchall():
    negative_list.append(row[0])

cur.execute("SELECT l.ID from lexicalunit l where (l.comment like '%+ m' or l.comment like '%+ s' or l.comment like '%+ m %' or l.comment like '%+ s %')")

positive_list=list()#cur.fetchall()
for row in cur.fetchall():
    positive_list.append(row[0])

#print 'n ',negative_list
#print 'p ',positive_list
#print 'd ', not_disamb_list

db.close()


clf=MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
relations=[] #wszystkie lub hiponimy hiperonimy antonimia wlasciwa
list_of_dicts=list()
list_of_polar=dict()
synsets_polar=dict()
path='/home/aleksandradolega/'



print path
base=BaseGraph()
base.unpickle(path+'merged_graph.xml.gz')
lu_graph=BaseGraph()
lu_graph.unpickle(path+'OUTPUT_GRAPHS_lu.xml.gz')
#print base._g.vertex_properties()
base._g.list_properties()
print 'paus'
lu_graph._g.list_properties()


#c=0
for n in base.all_nodes():
    non=False
    local_polar = list()
    for lu in n.synset.lu_set:

        if not list_of_polar.has_key(lu.lu_id):
            idL=lu.lu_id#str(lu.lu_id)+"L"

            #print 'idL: ',idL
            if idL in not_disamb_list:
                local_polar.append(0)
                continue
            if idL in positive_list:
                local_polar.append(1)
                list_of_polar[lu.lu_id]=1
                #print 'POS'
            elif idL in negative_list:
                local_polar.append(-1)
                list_of_polar[lu.lu_id] = -1
                #print 'NEG'
            else:
                local_polar.append(0)
                list_of_polar[lu.lu_id] = 0


    count=0
    for val in local_polar:
        if val==0:
            count+=1
    if non:
        print n.synset.synset_id,'--> ',local_polar


    if count<0.1*len(local_polar):
        polarity=sum(local_polar)
        if  polarity < 0:
            polarity=-1
        elif polarity > 0:
            polarity=1
        synsets_polar[n]=polarity

#print 'lop ',list_of_polar
pos=list()
neg=list()
#print 'sp ',len(synsets_polar)

for s in synsets_polar.keys():
    if synsets_polar[s]==1:
        pos.append(s)
    elif synsets_polar[s]==-1:
        neg.append(s)
print 'NEGATIVE '
for s in neg:
    print '============================='
    for ll in s.synset.lu_set:
        print ll.lemma
    print s.synset.synset_id, ' -> ',synsets_polar[s]
    print '*****************************'
print 'POSITIVE '
for s in pos:
    print '============================='
    for ll in s.synset.lu_set:
        print ll.lemma
    print s.synset.synset_id, ' -> ',synsets_polar[s]
print '*******************************'
print len(synsets_polar)
print 'NEG: ',len(neg)
print 'POS: ',len(pos)

lu_dict=dict()
MIN=-1000000
for node in lu_graph.all_nodes():
    current=node
    distance=0
    visited=list()
    queue=list()
    queue_level=dict()
    queue_level[current]=0
    while current.lu.lu_id not in list_of_polar:
        visited.append(current)
        for item in current.all_edges():
            if current==item.source():
                target=item.target()
                if target not in visited:
                    queue.append(target)
                    queue_level[target]=queue_level[current]+1
            if current==item.target():
                source=item.source()
                if source not in visited:
                    queue.append(source)
                    queue_level[source] = queue_level[current] + 1
        if len(queue)==0:
            distance=MIN
            print 'BREAK'
            break
        current=queue[0]
        queue.remove(current)

    if distance!=MIN:
        distance=queue_level[current]
    lu_dict[current.lu.lu_id]=distance

print 'LU DIC : ',lu_dict

for n in base.all_nodes():
    #c+=1
    #print n, ' syn : ',n.synset.synset_id, ', ',n.synset.lu_set
    dic=dict()


    #for nn in n.synset.lu_set:
    #    print '** ',nn.lu_id,' ', nn.lemma,' ', nn.pos,' ',nn.domain,' ', nn.variant
    for e in n.all_edges():
        if e.target()==n:
            s=0

        else: #source
            s=2
    #    xx=1
    #     if dic.has_key(e.rel):
    #         dic[e.rel]=dic[e.rel]+1
    # print ':: ',e.source().synset.synset_id,'---> ', e.target().synset.synset_id,' ',e, ' ',e.rel_id, ' ',e.rel
    # for n2 in e.source().synset.lu_set:
    #    print 'in source: ',n2.lemma,' , ',n2.lu_id
    # for n2 in e.target().synset.lu_set:
    #    print 'in target: ', n2.lemma, ' , ', n2.lu_id
    tuple_for_dict = (n.synset.synset_id, dic)
    #if c==10:
    #    break
x=set()

print x

#def search_polarization_level(node):
#    if node.lu_

def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    dataset_size = len(X_train)
    #X_train = X_train.reshape(dataset_size, -1)

    dataset_size2 = len(X_test)
    #X_test = X_test.reshape(dataset_size2, -1)

    #X_train=X_train[:50000]
    #y_train = y_train[:50000]
    #X_test = X_test[:100]
    #y_test = y_test[:100]
    print 's -> ',len(X_train),' ',len(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_neural(attributes, labels, data):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(attributes, labels)
    result = clf.predict(data)
    return result
    #print '>', clf.predict([[2., 2.], [-1., -2.]])

def create_conv(attributes, labels, data, results):
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
        input_shape=(None, 1, 28, 28),
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
        output_num_units=10,
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

def predict(data):
    result=clf.predict(data)
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