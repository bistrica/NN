from sklearn.naive_bayes import GaussianNB
import numpy

class Bayes(object):
    gnb=None
    graph=None
    propagator=None
    X_train=None
    Y_train=None

    def __init__(self,propagator):
        self.gnb = GaussianNB()
        self.propagator=propagator
        self.graph=propagator.GRAPH
        self.X_train = list()
        self.Y_train = list()

    def create_model(self):
        if self.X_train==[] and self.Y_train==[]:

            for pol in self.graph.list_of_polar.keys():
                vec, label = self.propagator.get_vector(self.graph.lu_nodes[pol])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)

                self.X_train.append(vec)
                self.Y_train.append(label)

        self.gnb.fit(self.X_train, self.Y_train)

    def append_training_item(self,vec, res):

        self.X_train.append(vec)
        self.Y_train.append(res)

    def predict(self,data):
        y_pred = self.gnb.predict(data)
        return y_pred