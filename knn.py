import numpy
from sklearn import neighbors
class KNN(object):

    knn=None
    propagator=None
    graph = None
    X_train = []

    Y_train = []
    n_neighbors=5
    weights='uniform'
    algorithm='auto'

    ALL_WEIGHTS=['uniform','distance']
    ALL_ALGORITHMS=['auto', 'ball_tree', 'kd_tree', 'brute']


    def __init__(self,propagator):
        if propagator.neighbours_number is not None:
            self.n_neighbors=propagator.neighbours_number
        if propagator.knn_weights in self.ALL_WEIGHTS:
            self.weights=propagator.knn_weights
        if propagator.knn_algorithm in self.ALL_ALGORITHMS:
            self.algorithm=propagator.knn_algorithm

        self.knn= neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors,weights=self.weights,algorithm=self.algorithm)

        self.propagator=propagator
        self.graph = propagator.GRAPH

    def create_model(self):

        if self.X_train == [] and self.Y_train == []:

            for pol in self.graph.list_of_polar.keys():
                vec, label = self.propagator.get_vector(self.graph.lu_nodes[pol])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)

                self.X_train.append(vec)
                self.Y_train.append(label)
        self.knn.fit(self.X_train,self.Y_train)

    def predict(self,item):
        return self.knn.predict(item)

    def append_training_item(self,vec,label):
        self.X_train.append(vec)
        self.Y_train.append(label)