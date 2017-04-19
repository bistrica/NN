from wosedon.basegraph import BaseGraph
from Neural import Neural
#from graph_reader import GraphReader
import copy
from summarizer import Finder
import numpy
import pickle

class Propagator(object):
    MANUAL=0
    NEURAL=1

    PERCENT=0.5
    REL_IDS=[]
    WEIGHTS=[]
    TYPE=0
    GRAPH=None
    DEPTH=1
    TRAINING_DEPTH=2
    LAYERS_UNITS=[32,16,8]

    network=None
    network_path=''

    rel_positive=dict()
    rel_negative=dict()
    rel_none=dict()
    rel_amb=dict()
    data_dic=None


    #[-8, 10, 11, 12, 62, 104, 141, 169, 244]
    #-8:synonimia, 12-antonimia, 10-hiponimia,11-hiperonimia, 62-syn.miedzyparadygmatyczna,104-antonimia wlasciwa,141-syn.miedzypar.,169-syn.mmiedzy,244-syn..miedzypar
    def __init__(self, type, known_data_dic, graph, depth, training_depth=2, percent=1.0, rel_ids=[-8,10,11,12,62,104,141,169,244, 13,14,15],weights=[], neural_layers=None, network=None, save_network=None):#,19,20,21,22,23,24,25,26,27,28,29,30], weights=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]):#15,2,2,-10,10,-4,10,10,10,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10]):#rel_ids=[-8], weights=[1]):#

        self.TYPE=type
        self.data_dic=known_data_dic
        self.REL_IDS=rel_ids
        self.WEIGHTS=weights
        self.GRAPH=graph
        self.DEPTH=depth
        self.TRAINING_DEPTH=training_depth
        self.PERCENT = percent
        if neural_layers is not None:
            self.LAYERS_UNITS=neural_layers
        if network is not None and network!='':
            self.network=pickle.load(open(network, "rb" ))
        if save_network is not None and save_network!='':
            self.network_path = save_network

    def create_neighbourhood(self, depth):
        finder = Finder()

        freq_map = finder.find_nearest_simple(self.GRAPH.lu_graph, self.GRAPH.list_of_polar, depth=depth,
                                              relations=self.get_relations())
        return freq_map

    def propagate(self):
        counter = self.DEPTH
        depth = 1
        if self.TYPE==self.MANUAL:
            counter = 0
            good_res = True
            while counter > 0 and good_res:
                counter -= 1
                dist_map = self.create_neighbourhood(depth+1)
                print 'dist_map map ', dist_map
                freq_set = list()

                for i in range(1, depth, 1):
                    freq_set.append(list())
                for k in dist_map.keys():
                    if dist_map[k] == 0:
                        continue
                    freq_set[dist_map[k] - 1].append(k)
                # sortowanie od najmniejszej liczby sasiadow
                for i in range(len(freq_set)):
                    freq_set[i] = sorted(freq_set[i], cmp=self.make_comparator(self.cmpValue), reverse=True)

                good_res = False
                for i in range(len(freq_set)):
                    for elem in freq_set[i]:
                        res = self.evaluate_node_percent(elem)
                        if res != -2:
                            good_res = True

        elif self.TYPE==self.NEURAL:
            if self.network is None:
                self.network = Neural(self,self.LAYERS_UNITS)
                X_train, X_test, Y_train, Y_test = self.network.create_data(0.98)
                self.network.create_neural()  # X[0], Y[0], X[1], Y[1])
                old_keys = copy.deepcopy(self.GRAPH.list_of_polar)

                training_counter=self.TRAINING_DEPTH
            else:
                training_counter=0


            while counter > 0:
                counter -= 1
                training_counter-=1
                dist_map = self.create_neighbourhood(depth + 1)
                for node in dist_map.keys():
                    if dist_map[node] == 0:
                        continue
                    vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                    vec = numpy.asarray(vec)
                    res = self.network.predict(vec)
                    self.network.append_training_item(vec, res)

                    self.GRAPH.list_of_polar[node.lu.lu_id] = res
                    node.lu.polarity = res
                if training_counter>0:
                    self.network.create_neural()

            print 'NEU RES, ', self.network.res
            if self.network_path!='' and self.network_path is not None:
                pickle.dump(self.network, self.network_path)

    def make_comparator(less_than):
        def compare(x, y):
            if less_than(x, y):
                return -1
            elif less_than(y, x):
                return 1
            else:
                return 0

        return compare

    def cmpValue(node1, node2):
        n1 = 0
        n2 = 0
        for n in node1.all_edges():
            n1 += 1
        for n in node2.all_edges():
            n2 += 1
        return n1 > n2

    def get_relations(self):
        return self.REL_IDS

    def get_vector(self,node):
        self.rel_positive = dict()
        self.rel_negative = dict()
        self.rel_none = dict()
        self.rel_amb = dict()

        for e in node.all_edges():
            if e.rel_id in self.REL_IDS:
                polarity=0
                scnd_node=None
                if e.source()==node:
                    scnd_node=e.target()

                else:
                    scnd_node=e.source()
                dic_to_update=dict()

                if self.data_dic.has_key(scnd_node.lu.lu_id):

                    polarity=self.data_dic[scnd_node.lu.lu_id]

                    if polarity>0:
                        dic_to_update = self.rel_positive

                    elif polarity<0:
                        dic_to_update = self.rel_negative

                    else:
                        dic_to_update = self.rel_amb

                else:
                   dic_to_update=self.rel_none
                if dic_to_update.has_key(e.rel_id):
                    dic_to_update[e.rel_id]+=1
                else:
                    dic_to_update[e.rel_id] = 1
        vector_p=list()
        vector_n = list()
        vector_a = list()
        vec_tuples=list()

        for rel in self.REL_IDS:
            if self.rel_positive.has_key(rel):
                vector_p.append(self.rel_positive[rel])
            else:
                vector_p.append(0)
            if self.rel_negative.has_key(rel):
                vector_n.append(self.rel_negative[rel])
            else:
                vector_n.append(0)
            if self.rel_amb.has_key(rel):
                vector_a.append(self.rel_amb[rel])
            else:
                vector_a.append(0)

            label=-10
            if self.data_dic.has_key(node.lu.lu_id):
                label=self.data_dic[node.lu.lu_id]

        vector_p.extend(vector_n)
        vector_p.extend(vector_a)
        #print 'v:',vector_p
        return (vector_p, label)

    def evaluate_node_percent(self,node):
        percent=self.PERCENT
        self.CN+=1
        self.rel_positive = dict()
        self.rel_negative = dict()
        self.rel_none = dict()
        self.rel_amb = dict()
        count=0
        none=0

        for e in node.all_edges():


            if e.rel_id in self.REL_IDS:
                count+=1

                polarity=0
                scnd_node=None
                if e.source()==node:
                    scnd_node=e.target()

                else:
                    scnd_node=e.source()
                dic_to_update=dict()

                if self.data_dic.has_key(scnd_node.lu.lu_id):

                    polarity=self.data_dic[scnd_node.lu.lu_id]

                    if polarity>0:
                        dic_to_update = self.rel_positive

                    elif polarity<0:
                        dic_to_update = self.rel_negative

                    else:
                        dic_to_update = self.rel_amb

                else:
                    none+=1
                    dic_to_update=self.rel_none

                if dic_to_update.has_key(e.rel_id):
                    dic_to_update[e.rel_id]+=1
                else:
                    dic_to_update[e.rel_id] = 1
        vector_p=list()
        vector_n = list()
        vector_a = list()
        vec_tuples=list()

        if count!=0 and none!=count and none<=percent*count:
            #if not(len(self.rel_positive)!=0 and len(self.rel_negative)!=0):
            for rel in self.REL_IDS:
                if self.rel_positive.has_key(rel):
                    vector_p.append(self.rel_positive[rel])
                else:
                    vector_p.append(0)
                if self.rel_negative.has_key(rel):
                    vector_n.append(self.rel_negative[rel])
                else:
                    vector_n.append(0)
                if self.rel_amb.has_key(rel):
                    vector_a.append(self.rel_amb[rel])
                else:
                    vector_a.append(0)
            pos = sum([a * b for a, b in zip(vector_p, self.WEIGHTS)])
            neg = sum([a * b for a, b in zip(vector_n, self.WEIGHTS)])
            amb = sum([a * b for a, b in zip(vector_a, self.WEIGHTS)])
            if pos < 0:
                if neg < 0:
                    n = neg
                    neg = -1 * pos
                    pos = -1 * n
                else:
                    neg += -1 * pos
                    pos=0
            if neg < 0:
                if pos < 0:
                    p = pos
                    pos = -1 * neg
                    neg = -1 * p
                else:
                    pos += -1 * neg
                    neg=0


            print 'vec ',vector_p,' -  ',vector_n, ' - ',vector_a
            print 'f Pos: ', pos, ' neg: ', neg, ', amb ', amb, "(", node.lu.lemma, ',', node.lu.variant, ' - '
            if self.data_dic.has_key(node.lu.lu_id):
                print '[',self.data_dic[node.lu.lu_id],']'

            if pos>neg and pos>amb:
                res=1
            elif neg>pos and neg>amb:
                res=-1
            elif amb>pos and amb>neg:
                res=0
            elif pos==neg and pos>amb:
                res=0
            elif amb==pos:
                res=1
            elif amb==neg:
                res=-1
            else:
                res=-2
            if res!=-2:
                self.data_dic[node.lu.lu_id]=res

            print res,' Pos: ', pos, ' neg: ', neg, ', amb ', amb, "(", node.lu.lemma, ',', node.lu.variant, ' - '#, \
            return res