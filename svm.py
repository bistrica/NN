from sklearn import svm
import numpy
import thread
from threading import Thread
import time

class SVM(object):

    kernel=None
    kernels=['linear','sigmoid','rbf','poly']
    graph = None
    svc=None
    propagator=None
    X_train = []
    Y_train = []


    def __init__(self,propagator):
        kernel='linear'
        if propagator.kernel in self.kernels:
            kernel=propagator.kernel
        self.kernel=kernel
        self.svc = svm.SVC(kernel=self.kernel, C=1.0)
        self.propagator=propagator
        self.graph = propagator.GRAPH

    def create_model(self):
        self.X_train = list()
        self.Y_train = list()

        ttt=time.time()
        if True:
            keys=self.graph.list_of_polar.keys()
            k1=list()
            k2=list()
            k3=list()
            for i in range(len(keys)/3):
                k1.append(keys[i])
            for i in range(len(keys)/3,len(keys)/3*2):
                k2.append(keys[i])
            for i in range(len(keys)/3*2,len(keys)):
                k3.append(keys[i])

            ret1=list()
            ret2 = list()
            ret3 = list()

            t1 = ThreadWithReturnValue(target=self.create_vectors, args=(k1,))
            t2 = ThreadWithReturnValue(target=self.create_vectors, args=(k2,))
            t3 = ThreadWithReturnValue(target=self.create_vectors, args=(k3,))
            #thread.start_new_thread(self.create_vectors, ())
            #thread.start_new_thread(self.create_vectors, (k2,ret2))
            #thread.start_new_thread(self.create_vectors, (k2,ret3))
            t1.start()
            t2.start()
            t3.start()


            ret1=t1.join()
            ret2=t2.join()
            ret3=t3.join()
            #print 'r1 ',ret1

            for tup in ret1:
                vec,label=tup
                self.X_train.append(vec)
                self.Y_train.append(label)
            for tup in ret2:
                vec,label=tup
                self.X_train.append(vec)
                self.Y_train.append(label)
            for tup in ret3:
                vec,label=tup
                self.X_train.append(vec)
                self.Y_train.append(label)


        if False:
            for pol in self.graph.list_of_polar.keys():

                vec, label = self.propagator.get_vector(self.graph.lu_nodes[pol])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)

                self.X_train.append(vec)
                self.Y_train.append(label)

        self.svc.fit(self.X_train,self.Y_train)


    def create_vectors(self,keys):
        vecs=list()
        labs=list()
        for i in keys:

            vec, label = self.propagator.get_vector(self.graph.lu_nodes[i])
            if vec is None:
                continue
            vec = numpy.asarray(vec)
            vecs.append(vec)
            labs.append(label)

        ret= zip(vecs,labs)

        return ret

    def predict(self,item):
        return self.svc.predict(item)

    def append_training_item(self,vec,label):
        self.X_train.append(vec)
        self.Y_train.append(label)

from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
        #self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return

