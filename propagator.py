
from Neural import Neural
import copy
from summarizer import Finder
import numpy
from bayes import Bayes
import pickle
from knn import KNN
from svm import SVM
from collections import OrderedDict

class Propagator(object):
    MANUAL=0
    NEURAL=1
    NEURAL_MULTIPLE=2
    BAYES=3
    KNN=4
    SVM=5

    PERCENT=0
    REL_IDS=[]
    WEIGHTS=[]
    TYPE=0
    GRAPH=None
    DEPTH=1
    TRAINING_DEPTH=2
    LAYERS_UNITS=[32,16,8]
    ALL_POS=[1,2,3,4]#verb,noun,adverb,adjective
    #MIN_PERCENT=0

    normalization=False
    network=None
    network_path=None
    new_lu_data_path=None
    chosen_pos=None
    kernel=None

    rel_positive=dict()
    rel_negative=dict()
    rel_none=dict()
    rel_amb=dict()
    data_dic=None


    #[-8, 10, 11, 12, 62, 104, 141, 169, 244]
    #-8:synonimia, 12-antonimia, 10-hiponimia,11-hiperonimia, 62-syn.miedzyparadygmatyczna,104-antonimia wlasciwa,141-syn.miedzypar.,169-syn.mmiedzy,244-syn..miedzypar
    def __init__(self, type, known_data_dic, graph, depth, training_depth=2, normalization=False, percent=0.0, rel_ids=[-8,10,11,12,62,104,141,169,244, 13,14,15],weights=[], neural_layers=None, network=None, save_network=None, save_new_lu_polarities=None, chosen_pos=None, kernel=None, neighbours_number=None, knn_algorithm=None, knn_weights=None):#,min_percent=0):#,19,20,21,22,23,24,25,26,27,28,29,30], weights=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]):#15,2,2,-10,10,-4,10,10,10,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10]):#rel_ids=[-8], weights=[1]):#

        self.TYPE=type
        self.data_dic=known_data_dic
        self.REL_IDS=rel_ids
        self.WEIGHTS=weights
        self.GRAPH=graph
        self.DEPTH=depth
        self.TRAINING_DEPTH=training_depth
        self.PERCENT = percent
        self.normalization=normalization
        self.chosen_pos=chosen_pos
        self.kernel=kernel
        self.knn_algorithm=knn_algorithm
        self.knn_weights=knn_weights
        self.neighbours_number=neighbours_number

        if neural_layers is not None:
            self.LAYERS_UNITS=neural_layers
        if network is not None:# and network!='':
            #if type==Propagator.NEURAL:
            self.network=pickle.load(open(network, "rb" ))
            #elif type==Propagator.NEURAL_MULTIPLE:
                #net_list = pickle.load(open(network, "rb"))
            #    self.network = pickle.load(open(network, "rb"))
        if save_network is not None:# and save_network!='':
            self.network_path = save_network
        if save_new_lu_polarities is not None:# and save_new_lu_polarities!='':
            self.new_lu_data_path=save_new_lu_polarities

    def create_neighbourhood(self, depth):
        finder = Finder()

        freq_map = finder.find_nearest_simple(self.GRAPH.lu_graph, self.GRAPH.list_of_polar, depth=depth,
                                              relations=self.get_relations())
        return freq_map

    def propagate(self):
        old_keys = copy.deepcopy(self.GRAPH.list_of_polar)

        if self.TYPE==self.MANUAL:
            self.propagate_manual()

        elif self.TYPE==self.NEURAL:
            self.propagate_neural()
        elif self.TYPE==self.NEURAL_MULTIPLE:
            self.propagate_neural_multiple()
        elif self.TYPE==self.BAYES:
            classifier=Bayes(self)
            self.propagate_classifier(classifier)
        elif self.TYPE==self.KNN:
            classifier=KNN(self)
            self.propagate_classifier(classifier)
        elif self.TYPE==self.SVM:
            classifier=SVM(self)
            self.propagate_classifier(classifier)

        if self.new_lu_data_path is not None:
            file = open(self.new_lu_data_path, 'wr+')

            self.data_dic=OrderedDict(sorted(self.data_dic.items(), key=lambda t: t[1]))
            #keys.sort()
            keys = self.data_dic.keys()
            for k in keys:
                if k not in old_keys.keys():
                    file.write(str(k) + ', ' + str(self.GRAPH.lu_nodes[k].lu.lemma) + ', ' + str(
                        self.GRAPH.lu_nodes[k].lu.variant) + ', ' + str(self.data_dic[k]) + '\n')

    def propagate_classifier(self,classifier):
        classifier.create_model()

        counter = self.DEPTH
        training_counter = self.TRAINING_DEPTH
        depth = 1

        while counter > 0:

            counter -= 1
            training_counter -= 1
            dist_map = self.create_neighbourhood(depth + 1)
            for node in dist_map.keys():
                if dist_map[node] == 0:
                    continue
                vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)
                res = classifier.predict(vec)

                self.GRAPH.list_of_polar[node.lu.lu_id] = res
                node.lu.polarity = res
                self.data_dic[node.lu.lu_id] = res
            if training_counter > 0:
                classifier.append_training_item(vec, res)
                classifier.create_model()



    def propagate_manual(self):
        counter = self.DEPTH
        depth = 1

        good_res = True
        while counter > 0 and good_res:
            counter -= 1
            dist_map = self.create_neighbourhood(depth + 1)

            freq_set = list()

            for i in range(1, depth+1, 1):
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


    def propagate_neural(self):

        counter = self.DEPTH
        depth = 1


        if self.network is None:
            self.network = Neural(self, self.LAYERS_UNITS)
            X_train, X_test, Y_train, Y_test = self.network.create_data(1.0)
            self.network.create_neural()

            training_counter = self.TRAINING_DEPTH
        else:
            training_counter = 0


        while counter > 0:

            counter -= 1
            training_counter -= 1
            dist_map = self.create_neighbourhood(depth + 1)

            for node in dist_map.keys():
                if dist_map[node] == 0:
                    continue
                vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)
                res = self.network.predict(vec)
                self.network.append_training_item(vec, res)

                self.GRAPH.list_of_polar[node.lu.lu_id] = res
                node.lu.polarity = res
                self.data_dic[node.lu.lu_id] = res
            if training_counter > 0:
                self.network.create_neural()

        if self.network_path != '' and self.network_path is not None:
            file = open(self.network_path, 'wr+')
            pickle.dump(self.network, file)

    def propagate_neural_multiple(self):

        counter = self.DEPTH
        depth = 1


        if self.network is None:
            self.network = list()
            if self.chosen_pos is None:
                self.chosen_pos=self.ALL_POS
            for pos in self.ALL_POS:
                if pos in self.chosen_pos:
                    network_pos = Neural(self, self.LAYERS_UNITS)
                    network_pos.create_data(1.0,pos)
                    network_pos.create_neural()
                else:
                    network_pos=None

                self.network.append(network_pos)
            #network_pos4=Neural(self, self.LAYERS_UNITS)
            #network_pos2=Neural(self, self.LAYERS_UNITS)
            #network_pos4.create_data(1.0)
            #network_pos2.create_data(1.0)
            #network_pos4.create_neural()
            #network_pos2.create_neural()

            #self.network=[network_pos2,network_pos4]

            training_counter = self.TRAINING_DEPTH
        else:
            print 'ELS'
            training_counter = 0


        while counter > 0:
            print 'WH'
            counter -= 1
            training_counter -= 1
            dist_map = self.create_neighbourhood(depth + 1)

            for node in dist_map.keys():
                if dist_map[node] == 0:
                    continue
                vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)

                if node.lu.pos in self.chosen_pos:
                    #print '> ',node.lu.pos-1,' ',len(self.network)
                    net=self.network[node.lu.pos-1]

                    res = net.predict(vec)
                    net.append_training_item(vec, res)
                else:
                    continue

                self.GRAPH.list_of_polar[node.lu.lu_id] = res
                node.lu.polarity = res
                self.data_dic[node.lu.lu_id] = res
            if training_counter > 0:
                for net in self.network:
                    if net is not None:
                        net.create_neural()


        #print 'NEU RES, ', self.network.res
        if self.network_path != '' and self.network_path is not None:
            file = open(self.network_path, 'wr+')
            pickle.dump(self.network, file)


    def make_comparator(self,less_than):
        def compare(x, y):
            if less_than(x, y):
                return -1
            elif less_than(y, x):
                return 1
            else:
                return 0

        return compare

    def cmpValue(self,node1, node2):
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
        self.rel_positive_strong = dict()
        self.rel_negative_strong = dict()
        self.rel_none = dict()
        self.rel_amb = dict()
        count=0

        for e in node.all_edges():
            #print 'e.r',e.rel_id
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
                        if polarity==10:
                            dic_to_update = self.rel_positive_strong

                    elif polarity<0:
                        dic_to_update = self.rel_negative
                        if polarity==-10:
                            dic_to_update = self.rel_negative_strong

                    else:
                        dic_to_update = self.rel_amb

                else:
                   dic_to_update=self.rel_none
                if dic_to_update.has_key(e.rel_id):
                    dic_to_update[e.rel_id]+=1
                else:
                    dic_to_update[e.rel_id] = 1
        vector_p=list()
        vector_ps = list()
        vector_n = list()
        vector_ns = list()
        vector_a = list()
        vec_tuples=list()
        for rel in self.REL_IDS:
            if self.rel_positive.has_key(rel):
                vector_p.append(self.rel_positive[rel])
            else:
                vector_p.append(0)
            if self.rel_positive_strong.has_key(rel):
                vector_ps.append(self.rel_positive_strong[rel])
            else:
                vector_ps.append(0)
            if self.rel_negative.has_key(rel):
                vector_n.append(self.rel_negative[rel])
            else:
                vector_n.append(0)
            if self.rel_negative_strong.has_key(rel):
                vector_ns.append(self.rel_negative_strong[rel])
            else:
                vector_ns.append(0)
            if self.rel_amb.has_key(rel):
                vector_a.append(self.rel_amb[rel])
            else:
                vector_a.append(0)

            label=-10
            if self.data_dic.has_key(node.lu.lu_id):
                label=self.data_dic[node.lu.lu_id]
        vector_p.extend(vector_ps)
        vector_p.extend(vector_n)
        vector_p.extend(vector_ns)
        vector_p.extend(vector_a)

        if count==0 or float(sum(vector_p))/float(count)<=self.PERCENT:
            return (None,None)

        if self.normalization:
            vec_sum=sum(vector_p)
            for i in range(len(vector_p)):
                vector_p[i]=float(vector_p[i])/vec_sum
        return (vector_p, label)

    def check_result(self,pos,pos_s,neg,neg_s,amb):
        max_val=max([pos,pos_s,neg,neg_s,amb])
        if (pos==max_val and neg==max_val) or (pos_s==max_val and neg_s==max_val) or (pos_s==max_val and neg==max_val) or (pos==max_val and neg_s==max_val):
            res=-2
        elif pos + pos_s > neg + neg_s:
            if pos==max_val and pos_s==max_val:
                if amb==max_val:
                    res=1
                else:
                    res=10
            else:
                if pos==max_val:
                    res=1
                elif pos_s==max_val:
                    res=10
        elif pos + pos_s < neg + neg_s:
            if neg==max_val and neg_s==max_val:
                if amb==max_val:
                    res=-1
                else:
                    res=-10
            else:
                if neg==max_val:
                    res=-1
                elif neg_s==max_val:
                    res=-10
        elif amb==max_val:
            if neg_s==max_val:
                res=-10
            elif neg==max_val:
                res=-1
            elif pos_s == max_val:
                res=10
            elif pos == max_val:
                res=1
        else:
            res=-2
        return res






    def evaluate_node_percent(self,node):
        percent=self.PERCENT

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
                        if polarity==10:
                            dic_to_update = self.rel_positive_strong

                    elif polarity<0:
                        dic_to_update = self.rel_negative
                        if polarity==-10:
                            dic_to_update = self.rel_negative_strong

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
        vector_ps = list()
        vector_n = list()
        vector_ns = list()
        vector_a = list()
        vec_tuples=list()

        if count!=0 and none!=count and (count-none)>=percent*count:

            for rel in self.REL_IDS:
                if self.rel_positive.has_key(rel):
                    vector_p.append(self.rel_positive[rel])
                else:
                    vector_p.append(0)
                if self.rel_positive_strong.has_key(rel):
                    vector_ps.append(self.rel_positive_strong[rel])
                else:
                    vector_ps.append(0)
                if self.rel_negative.has_key(rel):
                    vector_n.append(self.rel_negative[rel])
                else:
                    vector_n.append(0)
                if self.rel_negative_strong.has_key(rel):
                    vector_ns.append(self.rel_negative_strong[rel])
                else:
                    vector_ns.append(0)
                if self.rel_amb.has_key(rel):
                    vector_a.append(self.rel_amb[rel])
                else:
                    vector_a.append(0)
            pos = sum([a * b for a, b in zip(vector_p, self.WEIGHTS)])
            pos_s = sum([a * b for a, b in zip(vector_ps, self.WEIGHTS)])
            neg = sum([a * b for a, b in zip(vector_n, self.WEIGHTS)])
            neg_s = sum([a * b for a, b in zip(vector_ns, self.WEIGHTS)])
            amb = sum([a * b for a, b in zip(vector_a, self.WEIGHTS)])
            results=[pos,pos_s,neg,neg_s,amb]
            max_res=max(results)
            proper_win = results.count(max_res) == 1

            #if pos < 0:
            #    if neg < 0:
            #        n = neg
            #        neg = -1 * pos
            #        pos = -1 * n
            #    else:
            #        neg += -1 * pos
            #        pos=0
            #if neg < 0:
            #    if pos < 0:
            #        p = pos
            #        pos = -1 * neg
            #        neg = -1 * p
            #    else:
            #        pos += -1 * neg
            #        neg=0


            #print 'vec ',vector_p,' -  ',vector_n, ' - ',vector_a
            #print 'f Pos: ', pos, ' neg: ', neg, ', amb ', amb, "(", node.lu.lemma, ',', node.lu.variant, ' - '
            if self.data_dic.has_key(node.lu.lu_id):
                print '[',self.data_dic[node.lu.lu_id],']'

            if proper_win:
                if pos==max_res:
                    res=1
                if pos_s==max_res:
                    res=10
                if neg==max_res:
                    res=-1
                if neg_s==max_res:
                    res=-10
                if amb==max_res:
                    res=0
            else:
                res=self.check_result(pos,pos_s,neg,neg_s,amb)
                #res = -2



            if False:
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

            if res != -2:
                self.data_dic[node.lu.lu_id] = res

            #print res,' Pos: ', pos, ' neg: ', neg, ', amb ', amb, "(", node.lu.lemma, ',', node.lu.variant, ' - '#, \
            return res