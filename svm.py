from sklearn import svm
import numpy

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
        for pol in self.graph.list_of_polar.keys():
            vec, label = self.propagator.get_vector(self.graph.lu_nodes[pol])
            if vec is None:
                continue
            vec = numpy.asarray(vec)

            self.X_train.append(vec)
            self.Y_train.append(label)
        self.svc.fit(self.X_train,self.Y_train)

    def predict(self,item):
        return self.svc.predict(item)

    def append_training_item(self,vec,label):
        self.X_train.append(vec)
        self.Y_train.append(label)