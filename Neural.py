import numpy

from sklearn.neural_network import MLPClassifier



class Neural(object):

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


    #def create_mlp(self):
    #    mlp=self.build_mlp()
    #    print 'kel ',len(self.X_train),len(self.Y_train)
    #    mlp.fit(self.X_train,self.Y_train,200)
    #    preds = mlp.predict(self.X_test)
    #    cm = confusion_matrix(self.Y_test, preds)
    #    plt.matshow(cm)
    #    plt.title('Confusion matrix')
    #    plt.colorbar()
    #    plt.ylabel('True label')
    #    plt.xlabel('Predicted label')
    #    plt.show()

    def create_neural(self):#,attributes, labels, data, data_labels):
        print 'lenn ',len(self.X_train), len(self.Y_train), len(self.X_test), len(self.Y_test)
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=self.hidden_layers, random_state=1)#5,2

        self.clf.fit(self.X_train, self.Y_train)
        self.predict_test_set()


    def predict_test_set(self):
        ccc = 0
        for i in range(len(self.X_test)):

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
