
from Neural import Neural
import copy
from summarizer import Finder
import numpy
from bayes import Bayes
import pickle
from knn import KNN
from svm import SVM
from collections import OrderedDict
import thread
import sys
from threading import Thread
import time
from sklearn.externals import joblib

__all__ = ("error", "LockType", "start_new_thread", "interrupt_main", "exit", "allocate_lock", "get_ident", "stack_size", "acquire", "release", "locked")

class Propagator(object):
    MANUAL=0
    NEURAL=1
    NEURAL_MULTIPLE=2
    BAYES=3
    KNN=4
    SVM=5
    ENSEMBLE=6

    PERCENT=0
    REL_IDS=[]
    WEIGHTS=[]
    TYPE=0
    GRAPH=None
    DEPTH=1
    TRAINING_DEPTH=2
    LAYERS_UNITS=[32,16,8]
    LAYERS_UNITS_NM = [32, 16, 8]

    ALL_POS=[1,2,3,4]#verb,noun,adverb,adjective
    SAVE_TO_DB = False
    NORMALIZATION=False
    NETWORK=None
    NETWORK_PATH=None
    NEW_LU_DATA_PATH=None
    CHOSEN_POS=None
    KERNEL=None
    SAVE_SVM_PATH=None
    SVM_MODEL=None

    classifier=None
    rel_positive=dict()
    rel_negative=dict()
    rel_positive_strong = dict()
    rel_negative_strong = dict()
    rel_none=dict()
    rel_amb=dict()
    data_dic=None

    debug=False
    only_classify_list = None


    # [-8, 10, 11, 12, 62, 104, 141, 169, 244]
    # -8:synonimia, 12-antonimia, 10-hiponimia,11-hiperonimia, 62-syn.miedzyparadygmatyczna,104-antonimia wlasciwa,141-syn.miedzypar.,169-syn.mmiedzy,244-syn..miedzypar
    def __init__(self, type, known_data_dic, graph, depth, training_depth=2, normalization=False, percent=0.0,
                 rel_ids=[-8, 10, 11, 12, 62, 104, 141, 169, 244, 13, 14, 15], weights=[], neural_layers=None, neural_layers_multiple=None,
                 network=None, save_network=None, save_new_lu_polarities=None, chosen_pos=None, kernel=None,
                 neighbours_number=None, knn_algorithm=None, knn_weights=None, ensemble_path=None,
                 save_ensemble_path=None, svm_model=None, save_svm_path=None,
                 save_to_db=False,only_classify_list=None):
        self.TYPE = type
        self.data_dic = known_data_dic
        self.REL_IDS = rel_ids
        self.WEIGHTS = weights

        self.GRAPH = graph
        self.DEPTH = depth
        self.TRAINING_DEPTH = training_depth
        self.PERCENT = percent
        self.NORMALIZATION = normalization
        self.CHOSEN_POS = chosen_pos
        self.KERNEL = kernel
        self.knn_algorithm = knn_algorithm
        self.knn_weights = knn_weights
        self.neighbours_number = neighbours_number

        if neural_layers is not None:
            self.LAYERS_UNITS = neural_layers
        if neural_layers_multiple is not None:
            self.LAYERS_UNITS_NM = neural_layers_multiple

        if network is not None:
            self.NETWORK = pickle.load(open(network, "rb"))

        self.NETWORK_PATH = save_network

        self.NEW_LU_DATA_PATH = save_new_lu_polarities
        self.ensemble_path = ensemble_path
        self.save_ensemble_path = save_ensemble_path
        self.SAVE_SVM_PATH=save_svm_path
        self.SVM_MODEL=svm_model
        if svm_model is not None:
            self.SVM_MODEL = joblib.load(svm_model)
        self.SAVE_TO_DB = save_to_db
        self.only_classify_list=only_classify_list

    def create_ensemble(self):
        if self.ensemble_path is None:
            pr1 = Propagator(type=Propagator.SVM, known_data_dic=copy.deepcopy(self.GRAPH.list_of_polar), graph=self.GRAPH,
                             depth=self.DEPTH,
                             normalization=self.NORMALIZATION,
                             training_depth=self.TRAINING_DEPTH,
                             percent=self.PERCENT, rel_ids=self.REL_IDS,
                             kernel=self.KERNEL,only_classify_list=self.only_classify_list)
            pr2 = Propagator(type=Propagator.NEURAL_MULTIPLE, known_data_dic=copy.deepcopy(self.GRAPH.list_of_polar),
                             graph=self.GRAPH,
                             depth=self.DEPTH, normalization=self.NORMALIZATION,
                             training_depth=self.TRAINING_DEPTH,
                             percent=self.PERCENT, rel_ids=self.REL_IDS, neural_layers=self.LAYERS_UNITS_NM,

                             chosen_pos=self.CHOSEN_POS,only_classify_list=self.only_classify_list)
            pr3 = Propagator(type=Propagator.NEURAL, known_data_dic=copy.deepcopy(self.GRAPH.list_of_polar), graph=self.GRAPH,
                             depth=self.DEPTH,
                             training_depth=self.TRAINING_DEPTH, normalization=self.NORMALIZATION,
                             percent=self.PERCENT, rel_ids=self.REL_IDS, neural_layers=self.LAYERS_UNITS,only_classify_list=self.only_classify_list

                             )
        else:
            [pr1, pr2, pr3,svm_model] = joblib.load(self.ensemble_path)
            pr1.classifier=SVM()
            pr1.svm_model=svm_model

            pr1.TRAINING_DEPTH = 0
            pr2.TRAINING_DEPTH = 0
            pr3.TRAINING_DEPTH = 0
            pr1.DEPTH = self.DEPTH
            pr2.DEPTH = self.DEPTH
            pr3.DEPTH = self.DEPTH
            pr1.PERCENT = self.PERCENT
            pr2.PERCENT = self.PERCENT
            pr3.PERCENT = self.PERCENT

        self.create_multithread_ensemble(pr1, pr2, pr3)
        if self.save_ensemble_path is not None:
            joblib.dump([pr1, pr2, pr3,pr1.classifier.svc], self.save_ensemble_path)


    def create_multithread_ensemble(self,pr,pr2,pr3):

        try:
            t1=Thread(target=pr.propagate)
            t2=Thread(target=pr2.propagate)
            t3 = Thread(target=pr3.propagate)
            t1.start()
            t2.start()
            t3.start()

            t1.join()
            t2.join()
            t3.join()
        except:
            print 'Thread error: ',sys.exc_info()
            pr.propagate()
            pr2.propagate()
            pr3.propagate()

        pr.get_common_result(pr.data_dic, pr2.data_dic, pr3.data_dic)


    def create_neighbourhood(self, depth,polarized=None):
        finder = Finder()
        freq_map,polarized_items = finder.find_nearest_simple(self.GRAPH.lu_graph, self.GRAPH.list_of_polar, depth=depth,
                                              relations=self.get_relations(),polarized=polarized)
        return freq_map,polarized_items

    def get_common_result(self,data1,data2,data3):
        new_dic = dict()
        for k in data1.keys():

            if k in data2.keys() and k in data3.keys():
                val1 = data1[k]
                val2 = data2[k]
                val3 = data3[k]
                if not isinstance(val1,int):
                    data1[k]=data1[k].flatten()
                    val1 = data1[k][0]
                if not isinstance(val2, int):
                    data2[k] = data2[k].flatten()
                    val2 = data2[k][0]
                if not isinstance(val3, int):
                    data3[k] = data3[k].flatten()
                    val3 = data3[k][0]
                if val1 == val2 and val1 == val3:
                    new_dic[k] = data1[k]
                else:
                    polar_dic = dict()
                    polar_dic[-10] = 0
                    polar_dic[-1] = 0
                    polar_dic[0] = 0
                    polar_dic[1] = 0
                    polar_dic[10] = 0
                    polar_dic[val1] += 1
                    polar_dic[val2] += 1
                    polar_dic[val3] += 1

                    negative = polar_dic[-10] + polar_dic[-1]
                    positive = polar_dic[1] + polar_dic[10]
                    if negative > 0 and positive > 0:
                        continue
                    if negative > 0 and negative > polar_dic[0]:
                        if polar_dic[-10] > polar_dic[-1]:
                            new_dic[k] = -10
                        else:
                            new_dic[k] = -1
                    elif positive > 0 and positive > polar_dic[0]:
                        if polar_dic[10] > polar_dic[1]:
                            new_dic[k] = 10
                        else:
                            new_dic[k] = 1
                    else:
                        new_dic[k] = 0
        self.data_dic=new_dic
        return new_dic

    def propagate(self):
        old_keys = copy.deepcopy(self.GRAPH.list_of_polar)
        only_classify=dict()

        if self.only_classify_list is not None:
            self.DEPTH=self.TRAINING_DEPTH
            self.SAVE_TO_DB=False
            self.SAVE_MODIFIED_MERGED_GRAPH_PATH=None


        if self.TYPE==self.MANUAL:

            if self.only_classify_list is not None:
                self.PERCENT=0
                for item in self.only_classify_list:
                    res=self.evaluate_node_percent(self.GRAPH.get_node_by_id(item))
                    only_classify[item]=res
                print 'ONLY ',only_classify
            else:
                self.propagate_manual()

        elif self.TYPE==self.NEURAL:
            self.propagate_neural()
            if self.only_classify_list is not None:
                self.PERCENT=0
                for item in self.only_classify_list:
                    res=None
                    vec, label = self.get_vector(self.GRAPH.get_node_by_id(item))

                    print 'vec ',vec
                    if vec is not None:
                        vec = numpy.asarray(vec)
                        res = self.NETWORK.predict(vec)
                    only_classify[item]=res
                self.data_dic = only_classify
                print 'ONLY ',only_classify

        elif self.TYPE==self.NEURAL_MULTIPLE:
            self.propagate_neural_multiple()
            if self.only_classify_list is not None:
                self.PERCENT = 0
                for item in self.only_classify_list:
                    res=None
                    node=self.GRAPH.get_node_by_id(item)
                    vec, label = self.get_vector(node)
                    if vec is not None:
                        vec = numpy.asarray(vec)

                        if node.lu.pos in self.CHOSEN_POS:
                            net = self.NETWORK[node.lu.pos - 1]
                            res = net.predict(vec)

                    only_classify[item] = res
                self.data_dic = only_classify
                print 'ONLY ', only_classify
        elif self.TYPE==self.BAYES:
            classifier=Bayes(self)
            self.propagate_classifier(classifier)
        elif self.TYPE==self.KNN:
            classifier=KNN(self)
            self.propagate_classifier(classifier)
        elif self.TYPE==self.SVM:
            classifier=SVM(self)
            if self.SVM_MODEL is not None:
                classifier.set_svc(self.SVM_MODEL)
            self.classifier=classifier
            self.propagate_classifier(classifier)
        elif self.TYPE==self.ENSEMBLE:
            self.create_ensemble()

        if self.only_classify_list is not None:
            if self.TYPE in [Propagator.SVM,Propagator.BAYES,Propagator.KNN]:
                self.PERCENT = 0
                for item in self.only_classify_list:
                    res = None
                    vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                    if vec is not None:
                        vec = numpy.asarray(vec)
                        res = classifier.predict(vec)
                    only_classify[item] = res
                self.data_dic=only_classify
                print 'ONLY ', only_classify

            #self.save_propagated_data(old_keys)

        self.save_propagated_data(old_keys)




    def save_propagated_data(self,old_keys):
        if self.NEW_LU_DATA_PATH is not None or self.SAVE_TO_DB:
            update_dic=dict()
            vals=dict()
            prefix=''
            vals[-10]=prefix+'##ASENTI {mn}'
            vals[-1]=prefix+'##ASENTI: {sn}'
            vals[10] =prefix+ '##ASENTI: {mp}'
            vals[1] = prefix+'##ASENTI: {sp}'
            vals[0]=prefix+'##ASENTI: {a}'
            file = open(self.NEW_LU_DATA_PATH, 'wr+')

            self.data_dic=OrderedDict(sorted(self.data_dic.items(), key=lambda t: t[1]))
            keys = self.data_dic.keys()
            if self.NEW_LU_DATA_PATH is not None:
                if self.SAVE_TO_DB:
                    for k in keys:
                        if k not in old_keys.keys():
                            update_dic[k]=vals[self.data_dic[k][0]]
                            file.write(str(k) + ', ' + str(self.GRAPH.lu_nodes[k].lu.lemma) + ', ' + str(
                                self.GRAPH.lu_nodes[k].lu.variant) + ', ' + str(self.data_dic[k]) + '\n')
                else:
                    for k in keys:
                        if k not in old_keys.keys():
                            file.write(str(k) + ', ' + str(self.GRAPH.lu_nodes[k].lu.lemma) + ', ' + str(
                                self.GRAPH.lu_nodes[k].lu.variant) + ', ' + str(self.data_dic[k]) + '\n')
            else:
                for k in keys:
                    if k not in old_keys.keys():
                        update_dic[k] = vals[self.data_dic[k][0]]

            if len(update_dic)!=0:
                self.GRAPH.save_polarity_to_db(update_dic)



    def propagate_classifier(self,classifier):

        if (self.TYPE==Propagator.SVM):
            if self.SVM_MODEL is None:
                classifier.create_model()
            else:
                classifier.set_svc(self.SVM_MODEL)
        else:
            classifier.create_model()
        counter = self.DEPTH
        training_counter = self.TRAINING_DEPTH
        depth = 1

        polarized=None
        while counter > 0:

            counter -= 1
            training_counter -= 1
            dist_map,polarized = self.create_neighbourhood(depth + 1,polarized)
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
                polarized.append(node)
            if training_counter > 0:
                classifier.append_training_item(vec, res)
                classifier.create_model()


        if self.TYPE==Propagator.SVM and self.SAVE_SVM_PATH is not None:

            joblib.dump(self.classifier.svc, self.SAVE_SVM_PATH)

    def propagate_manual(self):
        counter = self.DEPTH
        depth = 1

        good_res = True
        polarized = None
        while counter > 0 and good_res:
            counter -= 1
            dist_map,polarized = self.create_neighbourhood(depth + 1,polarized)

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
                    if res is not None:
                        good_res = True




    def propagate_neural(self):

        counter = self.DEPTH
        depth = 1

        if self.NETWORK is None:
            self.NETWORK = Neural(self, self.LAYERS_UNITS)
            self.NETWORK.create_data(1.0)
            self.NETWORK.create_neural()
            training_counter = self.TRAINING_DEPTH
        else:
            training_counter = 0

        polarized = None
        while counter > 0:

            counter -= 1
            training_counter -= 1
            dist_map,polarized = self.create_neighbourhood(depth + 1,polarized)

            for node in dist_map.keys():
                if dist_map[node] == 0:
                    continue
                vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)
                res = self.NETWORK.predict(vec)
                self.NETWORK.append_training_item(vec, res)

                self.GRAPH.list_of_polar[node.lu.lu_id] = res
                node.lu.polarity = res
                self.data_dic[node.lu.lu_id] = res
                if polarized is None:
                    polarized=list()
                polarized.append(node)
            if training_counter > 0:
                self.NETWORK.create_neural()

        if self.NETWORK_PATH is not None:
            file = open(self.NETWORK_PATH, 'wr+')
            pickle.dump(self.NETWORK, file)

    def propagate_neural_multiple(self):

        counter = self.DEPTH
        depth = 1
        any_net=None

        if self.NETWORK is None:
            self.NETWORK = list()
            if self.CHOSEN_POS is None:
                self.CHOSEN_POS=self.ALL_POS

            for pos in self.ALL_POS:
                if pos in self.CHOSEN_POS:
                    network_pos = Neural(self, self.LAYERS_UNITS_NM)
                    any_net=network_pos
                    if self.debug:
                        network_pos.create_data(1.0,pos)
                        network_pos.create_neural()
                else:
                    network_pos=None

                self.NETWORK.append(network_pos)


            if not self.debug and any_net is not None:
                data_lists=any_net.create_data_lists(self.CHOSEN_POS)

                for i in range(len(data_lists)):
                    if self.NETWORK[i] is not None:
                        self.NETWORK[i].set_data(data_lists[i])
                        self.NETWORK[i].create_neural()
            training_counter = self.TRAINING_DEPTH
        else:
            training_counter = 0

        polarized=list()
        while counter > 0:

            counter -= 1
            training_counter -= 1
            dist_map,polarized = self.create_neighbourhood(depth + 1,polarized)

            for node in dist_map.keys():
                if dist_map[node] == 0:
                    continue
                vec, label = self.get_vector(self.GRAPH.lu_nodes[node.lu.lu_id])
                if vec is None:
                    continue
                vec = numpy.asarray(vec)
                #if node.lu.lu_id==89353:
                #    print 'NODE ',node,node.lu,node.lu.pos
                if node.lu.pos in self.CHOSEN_POS:
                    net=self.NETWORK[node.lu.pos - 1]

                    res = net.predict(vec)
                    polarized.append(node)
                    net.append_training_item(vec, res)
                else:
                    #if node.lu.lu_id == 89353:
                    #    print 'conti'
                    continue

                self.GRAPH.list_of_polar[node.lu.lu_id] = res
                node.lu.polarity = res

                self.data_dic[node.lu.lu_id] = res

            #print 'data dic' ,self.data_dic

            if training_counter > 0:
                for net in self.NETWORK:
                    if net is not None:
                        net.create_neural()

        if self.NETWORK_PATH != '' and self.NETWORK_PATH is not None:
            file = open(self.NETWORK_PATH, 'wr+')
            pickle.dump(self.NETWORK, file)


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

        rel_positive = dict()
        rel_negative = dict()
        rel_positive_strong = dict()
        rel_negative_strong = dict()
        rel_none = dict()
        rel_amb = dict()
        count=0

        for e in node.all_edges():

            if e.rel_id in self.REL_IDS:
                count+=1

                if e.source()==node:
                    scnd_node=e.target()

                else:
                    scnd_node=e.source()

                if self.data_dic.has_key(scnd_node.lu.lu_id):

                    polarity=self.data_dic[scnd_node.lu.lu_id]

                    if polarity>0:
                        dic_to_update = rel_positive
                        if polarity==10:
                            dic_to_update = rel_positive_strong

                    elif polarity<0:
                        dic_to_update = rel_negative
                        if polarity==-10:
                            dic_to_update = rel_negative_strong

                    else:
                        dic_to_update = rel_amb

                else:
                   dic_to_update=rel_none
                if dic_to_update.has_key(e.rel_id):
                    dic_to_update[e.rel_id]+=1
                else:
                    dic_to_update[e.rel_id] = 1
        vector_p=list()
        vector_ps = list()
        vector_n = list()
        vector_ns = list()
        vector_a = list()

        for rel in self.REL_IDS:
            rel=int(rel)
            if rel_positive.has_key(rel):
                vector_p.append(rel_positive[rel])
            else:
                vector_p.append(0)
            if rel_positive_strong.has_key(rel):
                vector_ps.append(rel_positive_strong[rel])
            else:
                vector_ps.append(0)
            if rel_negative.has_key(rel):
                vector_n.append(rel_negative[rel])
            else:
                vector_n.append(0)
            if rel_negative_strong.has_key(rel):
                vector_ns.append(rel_negative_strong[rel])
            else:
                vector_ns.append(0)
            if rel_amb.has_key(rel):
                vector_a.append(rel_amb[rel])
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

        if self.NORMALIZATION:
            vec_sum=sum(vector_p)
            for i in range(len(vector_p)):
                vector_p[i]=float(vector_p[i])/vec_sum
        return (vector_p, label)

    def check_result(self,pos,pos_s,neg,neg_s,amb):
        max_val=max([pos,pos_s,neg,neg_s,amb])
        res=None
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
            res=None
        return res

    def evaluate_node_percent(self,node):
        percent=self.PERCENT

        self.rel_positive = dict()
        self.rel_negative = dict()
        self.rel_positive_strong = dict()
        self.rel_negative_strong = dict()
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
            if self.WEIGHTS is None or len(self.WEIGHTS)==0:
                self.WEIGHTS=list()
                for i in vector_p:
                    self.WEIGHTS.append(1)
            pos = sum([a * b for a, b in zip(vector_p, self.WEIGHTS)])
            pos_s = sum([a * b for a, b in zip(vector_ps, self.WEIGHTS)])
            neg = sum([a * b for a, b in zip(vector_n, self.WEIGHTS)])
            neg_s = sum([a * b for a, b in zip(vector_ns, self.WEIGHTS)])
            amb = sum([a * b for a, b in zip(vector_a, self.WEIGHTS)])
            results=[pos,pos_s,neg,neg_s,amb]
            max_res=max(results)
            proper_win = results.count(max_res) == 1

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

            if res is not None:
                self.data_dic[node.lu.lu_id] = res

            return res

