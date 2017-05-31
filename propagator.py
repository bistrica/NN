
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
    ALL_POS=[1,2,3,4]#verb,noun,adverb,adjective
    #MIN_PERCENT=0

    normalization=False
    network=None
    network_path=None
    new_lu_data_path=None
    chosen_pos=None
    kernel=None
    save_svm_path=None
    svm_model=None
    classifier=None

    rel_positive=dict()
    rel_negative=dict()
    rel_none=dict()
    rel_amb=dict()
    data_dic=None

    debug=False
    save_to_db=False

    # [-8, 10, 11, 12, 62, 104, 141, 169, 244]
    # -8:synonimia, 12-antonimia, 10-hiponimia,11-hiperonimia, 62-syn.miedzyparadygmatyczna,104-antonimia wlasciwa,141-syn.miedzypar.,169-syn.mmiedzy,244-syn..miedzypar
    def __init__(self, type, known_data_dic, graph, depth, training_depth=2, normalization=False, percent=0.0,
                 rel_ids=[-8, 10, 11, 12, 62, 104, 141, 169, 244, 13, 14, 15], weights=[], neural_layers=None,
                 network=None, save_network=None, save_new_lu_polarities=None, chosen_pos=None, kernel=None,
                 neighbours_number=None, knn_algorithm=None, knn_weights=None, ensemble_path=None,
                 save_ensemble_path=None, svm_model=None, save_svm_path=None,
                 save_to_db=False):  # ,min_percent=0):#,19,20,21,22,23,24,25,26,27,28,29,30], weights=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]):#15,2,2,-10,10,-4,10,10,10,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10]):#rel_ids=[-8], weights=[1]):#

        self.TYPE = type
        self.data_dic = known_data_dic
        self.REL_IDS = rel_ids
        self.WEIGHTS = weights
        print 'weights ',weights
        self.GRAPH = graph
        self.DEPTH = depth
        self.TRAINING_DEPTH = training_depth
        self.PERCENT = percent
        self.normalization = normalization
        self.chosen_pos = chosen_pos
        self.kernel = kernel
        self.knn_algorithm = knn_algorithm
        self.knn_weights = knn_weights
        self.neighbours_number = neighbours_number

        if neural_layers is not None:
            self.LAYERS_UNITS = neural_layers
        if network is not None:  # and network!='':
            # if type==Propagator.NEURAL:
            self.network = pickle.load(open(network, "rb"))
            # elif type==Propagator.NEURAL_MULTIPLE:
            # net_list = pickle.load(open(network, "rb"))
            #    self.network = pickle.load(open(network, "rb"))
        # if save_network is not None:# and save_network!='':
        self.network_path = save_network
        # if save_new_lu_polarities is not None:# and save_new_lu_polarities!='':
        self.new_lu_data_path = save_new_lu_polarities
        self.ensemble_path = ensemble_path
        self.save_ensemble_path = save_ensemble_path
        self.save_svm_path=save_svm_path
        self.svm_model=svm_model
        if svm_model is not None:
            self.svm_model = joblib.load(svm_model)#open(svm_model, "rb"))
        self.save_to_db = save_to_db

    def create_ensemble(self):
        if self.ensemble_path is None:
            pr1 = Propagator(type=Propagator.SVM, known_data_dic=copy.deepcopy(self.GRAPH.list_of_polar), graph=self.GRAPH,
                             depth=self.DEPTH,
                             normalization=self.normalization,
                             training_depth=self.TRAINING_DEPTH,
                             percent=self.PERCENT, rel_ids=self.REL_IDS,
                             kernel=self.kernel)
            pr2 = Propagator(type=Propagator.NEURAL_MULTIPLE, known_data_dic=copy.deepcopy(self.GRAPH.list_of_polar),
                             graph=self.GRAPH,
                             depth=self.DEPTH, normalization=self.normalization,
                             training_depth=self.TRAINING_DEPTH,
                             percent=self.PERCENT, rel_ids=self.REL_IDS, neural_layers=self.LAYERS_UNITS,

                             chosen_pos=self.chosen_pos)
            pr3 = Propagator(type=Propagator.NEURAL, known_data_dic=copy.deepcopy(self.GRAPH.list_of_polar), graph=self.GRAPH,
                             depth=self.DEPTH,
                             training_depth=self.TRAINING_DEPTH, normalization=self.normalization,
                             percent=self.PERCENT, rel_ids=self.REL_IDS, neural_layers=self.LAYERS_UNITS

                             )
        else:
            [pr1, pr2, pr3,svm_model] = joblib.load(self.ensemble_path)
            pr1.classifier=SVM()
            #classifier#to check
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
            #file=open(self.save_ensemble_path, 'wr+')
            joblib.dump([pr1, pr2, pr3,pr1.classifier.svc], self.save_ensemble_path)


    def create_multithread_ensemble(self,pr,pr2,pr3):

        try:

            print '1'
            t1=Thread(target=pr.propagate)
            #thread.start_new_thread(pr.propagate,())
            print '2'
            t2=Thread(target=pr2.propagate)
            #thread.start_new_thread(pr2.propagate,())
            print '3'
            t3 = Thread(target=pr3.propagate)
            t1.start()
            t2.start()
            t3.start()

            t1.join()
            t2.join()
            t3.join()
           # thread.start_new_thread(pr3.propagate,())
        except:
            print 'Thread error: ',sys.exc_info()
            pr.propagate()
            pr2.propagate()
            pr3.propagate()
        #    print ("Error: unable to start thread")


        pr.get_common_result(pr.data_dic, pr2.data_dic, pr3.data_dic)
        #new_lu_data_path=path







    def create_neighbourhood(self, depth,polarized=None):
        #if polarized is None:#len(polarized)==0:
        #    polarized=list()#None
        finder = Finder()
        t=time.time()

        print 'finder'
        freq_map,polarized_items = finder.find_nearest_simple(self.GRAPH.lu_graph, self.GRAPH.list_of_polar, depth=depth,
                                              relations=self.get_relations(),polarized=polarized)
        t=time.time()-t
        print 'Time CN ',t
        return freq_map,polarized_items

    def get_common_result(self,data1,data2,data3):
        new_dic = dict()
        for k in data1.keys():

            if k in data2.keys() and k in data3.keys():
                #print 'k: ', k, data1[k], data2[k], data3[k]
                val1 = data1[k]
                val2 = data2[k]
                val3 = data3[k]
                print '[k]',data1[k]
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
        return new_dic

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
            if self.svm_model is not None:
                classifier.set_svc(self.svm_model)
            self.classifier=classifier
            self.propagate_classifier(classifier)
        #elif self.TYPE==self.ENSEMBLE:
        #    data0 = copy.deepcopy(self.data_dic)
        #    classifier=SVM(self)
        #    self.propagate_classifier(classifier)
        #    self.GRAPH.list_of_polar=copy.deepcopy(old_keys)
        #    data1 = copy.deepcopy(self.data_dic)
        #    self.data_dic=copy.deepcopy(data0)
        #    self.propagate_neural_multiple()
        #    self.GRAPH.list_of_polar = copy.deepcopy(old_keys)
        #    data2 = copy.deepcopy(self.data_dic)
        #    self.data_dic = copy.deepcopy(data0)
        #    self.network=None
        #    self.propagate_neural()
        #    data3 = copy.deepcopy(self.data_dic)
        #    new_dic=self.get_common_result(data1,data2,data3)
        #    print 'new dic ',new_dic
        #    self.data_dic=new_dic
        elif self.TYPE==self.ENSEMBLE:
            self.create_ensemble()
        self.save_propagated_data(old_keys)


    def save_propagated_data(self,old_keys):
        if self.new_lu_data_path is not None or self.save_to_db:
            update_dic=dict()
            vals=dict()
            prefix='ENS'
            vals[-10]=prefix+'{Auto: -m}'
            vals[-1]=prefix+'{Auto: -s}'
            vals[10] =prefix+ '{Auto: +m}'
            vals[1] = prefix+'{Auto: +s}'
            vals[0]=prefix+'{Auto: am0}'
            file = open(self.new_lu_data_path, 'wr+')

            self.data_dic=OrderedDict(sorted(self.data_dic.items(), key=lambda t: t[1]))
                #keys.sort()
            keys = self.data_dic.keys()
            if self.new_lu_data_path is not None:
                if self.save_to_db:
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
        print 'propagate'
        if (self.TYPE==Propagator.SVM):# and self.svm_model is None ):#((self.TYPE==Propagator.ENSEMBLE and self.ensemble_path is None) or (self.TYPE==Propagator.SVM and self.svm_model is None )):
            if self.svm_model is None:
                print 'model'
                classifier.create_model()
            else:
                classifier.set_svc(self.svm_model)
        else:
            classifier.create_model()
        print 'after model'
        counter = self.DEPTH
        training_counter = self.TRAINING_DEPTH
        depth = 1

        polarized=None
        while counter > 0:
            print 'while ',counter

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

        if self.TYPE==Propagator.SVM and self.save_svm_path is not None:

            joblib.dump(self.classifier.svc, self.save_svm_path)




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


        if self.network is None:
            print 'NN None!'
            self.network = Neural(self, self.LAYERS_UNITS)
            X_train, X_test, Y_train, Y_test = self.network.create_data(1.0)
            self.network.create_neural()

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
                res = self.network.predict(vec)
                self.network.append_training_item(vec, res)

                self.GRAPH.list_of_polar[node.lu.lu_id] = res
                node.lu.polarity = res
                self.data_dic[node.lu.lu_id] = res
                if polarized is None:
                    polarized=list()
                polarized.append(node)
            if training_counter > 0:
                self.network.create_neural()

        if self.network_path is not None:#self.network_path != '' and
            file = open(self.network_path, 'wr+')
            pickle.dump(self.network, file)

    def propagate_neural_multiple(self):

        counter = self.DEPTH
        depth = 1
        t=time.time()
        any_net=None

        if self.network is None:
            print 'NM None!'
            self.network = list()
            if self.chosen_pos is None:
                self.chosen_pos=self.ALL_POS

            for pos in self.ALL_POS:
                if pos in self.chosen_pos:
                    network_pos = Neural(self, self.LAYERS_UNITS)
                    any_net=network_pos
                    if self.debug:
                        network_pos.create_data(1.0,pos)
                        network_pos.create_neural()
                else:
                    network_pos=None

                self.network.append(network_pos)
            t = time.time()-t
            print 'Time AP ',t
            t = time.time()

            if not self.debug and any_net is not None:
                data_lists=any_net.create_data_lists(self.chosen_pos)

                t = time.time() - t
                print 'Time CDL ', t
                t = time.time()
                for i in range(len(data_lists)):
                    if self.network[i] is not None:
                        self.network[i].set_data(data_lists[i])
                        self.network[i].create_neural()

            #network_pos4=Neural(self, self.LAYERS_UNITS)
            #network_pos2=Neural(self, self.LAYERS_UNITS)
            #network_pos4.create_data(1.0)
            #network_pos2.create_data(1.0)
            #network_pos4.create_neural()
            #network_pos2.create_neural()

            #self.network=[network_pos2,network_pos4]

            training_counter = self.TRAINING_DEPTH
        else:

            training_counter = 0

        t=time.time()-t
        print 'Time PNM ',t

        polarized=list()
        while counter > 0:
            t=time.time()

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

                if node.lu.pos in self.chosen_pos:
                    #print '> ',node.lu.pos-1,' ',len(self.network)
                    net=self.network[node.lu.pos-1]

                    res = net.predict(vec)
                    polarized.append(node)
                    #polarized=None
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
            t=time.time()-t
            print 'Time WHILE ',counter,t


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

        rel_positive = dict()
        rel_negative = dict()
        rel_positive_strong = dict()
        rel_negative_strong = dict()
        rel_none = dict()
        rel_amb = dict()
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
        vec_tuples=list()
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
                #print 'rel ',rel,self.rel_negative
                #print 'rel ', rel
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

        if self.normalization:
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

            if res is not None:
                self.data_dic[node.lu.lu_id] = res

            #print res,' Pos: ', pos, ' neg: ', neg, ', amb ', amb, "(", node.lu.lemma, ',', node.lu.variant, ' - '#, \
            return res

