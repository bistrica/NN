
from wosedon.basegraph import BaseGraph

import MySQLdb
from sklearn.neural_network import MLPClassifier
from summarizer import Finder
from propagator import Propagator
from Neural import NeuralNet
import numpy
import time
import copy

class GraphReader(object):
    finder = Finder()
    lu_synset_dic = dict()
    synsets = list()
    not_disamb_list = list()
    positive_list = list()  # cur.fetchall()
    negative_list = list()  # cur.fetchall()
    amb_list = list()
    list_of_dicts = list()
    list_of_polar = dict()
    synsets_polar = dict()
    path = '/home/aleksandradolega/'
    lu_graph_path = ''
    merged_graph_path = ''

    polar_nodes = list()
    synsets_polar = dict()
    lu_nodes = dict()
    lu_graph=BaseGraph
    base = BaseGraph()

    def __init__(self, lu_graph_path, merged_graph_path=None, host=None, user=None, passw=None, db_name=None, rel_to_add=[10,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30]):
        self.added_relations=rel_to_add
        self.lu_graph_path = lu_graph_path
        if host is not None:
            t=time.time()
            self.get_data_from_db(host, user, passw, db_name)
            t = time.time() - t
            print 'Time0 ', t
            t = time.time()
        if merged_graph_path is not None:


            self.merged_graph_path = merged_graph_path

            self.base = BaseGraph()
            self.base.unpickle(self.merged_graph_path)  # path + 'merged_graph.xml.gz') path = '/home/aleksandradolega/'


        self.lu_graph = BaseGraph()
        t=time.time()
        self.lu_graph.unpickle(self.lu_graph_path)  # path + 'OUTPUT_GRAPHS_lu.xml.gz')
        t=time.time()-t
        print 'Time1 ',t
        t = time.time()
        if merged_graph_path is None:
            self.create_map_lu_node()
            t = time.time() - t
            print 'Time2 ', t
            t = time.time()
            self.create_lu_polar_list()
            t = time.time() - t
            print 'Time3 ', t
            #t = time.time()

    #def __init__(self, final_graph_path):
    #    self.lu_graph = BaseGraph()
    #    self.lu_graph.unpickle(final_graph_path)

    def get_data_from_db(self, host, user, passw, db_name):
        db = MySQLdb.connect(host=host,  # "localhost",    # your host, usually localhost
                             user=user,  # "root",         # your username
                             passwd=passw,  # "toor",  # your password
                             db=db_name)  # "wordTEST")        # name of the data base

        cur = db.cursor()

        cur.execute("SELECT l.ID from lexicalunit l where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %') and  (l.comment like '%+ m' or l.comment like '%+ s' or l.comment like '%+ m %' or l.comment like '%+ s %')")

            #"SELECT l.ID from lexicalunit l join lexicalunit l2 on l.lemma=l2.lemma where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %') and  (l2.comment like '%+ m' or l2.comment like '%+ s' or l2.comment like '%+ m %' or l2.comment like '%+ s %')")

        for row in cur.fetchall():
            self.not_disamb_list.append(row[0])

        cur.execute(
            "SELECT l.ID from lexicalunit l where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %')")

        for row in cur.fetchall():
            self.negative_list.append(row[0])

        cur.execute(
            "SELECT l.ID from lexicalunit l where (l.comment like '%+ m' or l.comment like '%+ s' or l.comment like '%+ m %' or l.comment like '%+ s %')")

        for row in cur.fetchall():
            self.positive_list.append(row[0])

        cur.execute(
            "SELECT l.ID from lexicalunit l where l.comment like '% amb %'")

        for row in cur.fetchall():
            self.amb_list.append(row[0])

        print 'n ', len(self.negative_list)
        print 'p ', len(self.positive_list)
        # print 'd ', not_disamb_list


        cur.execute("SELECT LEX_ID, SYN_ID FROM unitandsynset")

        syns_map = dict()

        for row in cur.fetchall():
            if syns_map.has_key(row[1]):
                syns_map[row[1]].append(row[0])

            else:
                lex = list()
                lex.append(row[0])
                syns_map[row[1]] = lex

            self.lu_synset_dic[row[0]] = syns_map[row[1]]

        print 'SYNS M ', len(syns_map), ' , ', len(self.lu_synset_dic)
        db.close()

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    relations = []  # wszystkie lub hiponimy hiperonimy antonimia wlasciwa

    def print_pos_neg(self, synsets_polar):
        # print 'lop ',list_of_polar
        pos = list()
        neg = list()
        # print 'sp ',len(synsets_polar)

        for s in synsets_polar.keys():
            if synsets_polar[s] == 1:
                pos.append(s)
            elif synsets_polar[s] == -1:
                neg.append(s)
        print 'NEGATIVE '
        for s in neg:
            print '============================='
            for ll in s.synset.lu_set:
                print ll.lemma
            print s.synset.synset_id, ' -> ', synsets_polar[s]
            print '*****************************'
        print 'POSITIVE '
        for s in pos:
            print '============================='
            for ll in s.synset.lu_set:
                print ll.lemma
            print s.synset.synset_id, ' -> ', synsets_polar[s]
        print '*******************************'
        print len(synsets_polar)
        print 'NEG: ', len(neg)
        print 'POS: ', len(pos)

    # base._g.list_properties()
    # print 'paus'


    # lu_graph._g.list_properties()


    # c=0
    def create_map_lu_node(self):
        for node in self.lu_graph.all_nodes():
            self.lu_nodes[node.lu.lu_id] = node
            #  if not lu_synset_dic.has_key(lu.lu_id):
            #      lu_synset_dic[lu.lu_id] = n.synset  # .lu_id]=n.synset

    def create_lu_polar_list(self):

        for n in self.lu_graph.all_nodes():
            if True:#not self.list_of_polar.has_key(n.lu.lu_id):
                idL = n.lu.lu_id  # str(lu.lu_id)+"L"


                if idL in self.not_disamb_list:
                    #self.list_of_polar[n.lu.lu_id] = 0
                    continue
                elif idL in self.positive_list:

                    self.list_of_polar[n.lu.lu_id] = 1
                    # print 'POS'
                elif idL in self.negative_list:

                    self.list_of_polar[n.lu.lu_id] = -1
                    # print 'NEG'
                elif idL in self.amb_list:

                    self.list_of_polar[n.lu.lu_id] = 0
                #else:
                    #self.list_of_polar[n.lu.lu_id] = -2
                    # list_of_polar[lu.lu_id] = 0 #zakom. do testu rozpiecia



    def create_lu_syn_polar_list(self, percent=0.5):
        cc = 0


        for n in self.base.all_nodes():
            non = False
            local_polar = list()
            # synsets.append(n.synset)

            for lu in n.synset.lu_set:
                cc += 1
                # print '> ',lu.lu_id
                # if not lu_synset_dic.has_key(lu.lu_id):
                #    lu_synset_dic[lu.lu_id]=n.synset#.lu_id]=n.synset

                if not self.list_of_polar.has_key(lu.lu_id):
                    idL = lu.lu_id  # str(lu.lu_id)+"L"

                    # print 'idL: ',idL
                    if idL in self.not_disamb_list:
                        local_polar.append(0)
                        continue
                    if idL in self.positive_list:
                        local_polar.append(1)
                        self.list_of_polar[lu.lu_id] = 1
                        # print 'POS'
                    elif idL in self.negative_list:
                        local_polar.append(-1)
                        self.list_of_polar[lu.lu_id] = -1
                        # print 'NEG'
                    else:
                        local_polar.append(0)
                        # list_of_polar[lu.lu_id] = 0 #zakom. do testu rozpiecia

            count = 0
            negative = False
            positive = False
            for val in local_polar:
                if val < 0:
                    negative = True
                if val > 0:
                    positive = True
                if val == 0:
                    count += 1
            if non:
                print n.synset.synset_id, '--> ', local_polar

        print 'CCC ', cc
        # if count<0.1*len(local_polar):
        #   polarity=sum(local_polar)
        #  if  polarity < 0:
        #      polarity=-1
        #  elif polarity > 0:
        #      polarity=1
        #  synsets_polar[n]=polarity

        if positive != negative and count <= percent * len(local_polar):
            polarity = sum(local_polar)
            if polarity < 0:
                polarity = -1
            elif polarity > 0:
                polarity = 1
            self.synsets_polar[n] = polarity


    def append_synset_relations(self):
        edges=list()
        for synset_edge in self.base.all_edges():
            if synset_edge.rel_id in self.added_relations:
                source_lu_set=synset_edge.source().synset.lu_set
                target_lu_set = synset_edge.target().synset.lu_set
                for source_lu in source_lu_set:
                    for target_lu in target_lu_set:
                        if self.lu_nodes.has_key(source_lu.lu_id) and self.lu_nodes.has_key(target_lu.lu_id):
                        #print ':;; ',source_lu.lu_id, source_lu.lemma,source_lu.variant
                            edges.append((self.lu_nodes[source_lu.lu_id],self.lu_nodes[target_lu.lu_id],synset_edge.rel_id))

        print ' xxsr ', len(edges)
        for edge in edges:
                # print 'edge ',edge
            self.lu_graph.add_edge(edge[0], edge[1], [['rel_id', edge[2]]], True)

    def read_graph(self):
        self.create_lu_syn_polar_list()
        self.create_map_lu_node()

        self.append_synonymy_edges()
        print 'APPENDING RELATIONS'
        self.append_synset_relations()
        print 'STOP APPENDING RELATIONS'


    def save_graph(self, path):
        #output = open(path, 'wb')

        # Pickle dictionary using protocol 0.
        #pickle.dump(self.lu_, output)

        # Pickle the list using the highest protocol available.
        #pickle.dump(selfref_list, output, -1)

        #output.close()
        self.lu_graph.pickle(path)





    def append_synonymy_edges(self):
        new_edges = list()
        a=0
        edges=set()
        for node in self.lu_graph.all_nodes():
            if self.lu_synset_dic.has_key(node.lu.lu_id):
                synonyms = self.lu_synset_dic[node.lu.lu_id]

                for synonym in synonyms:
                    if synonym==node.lu.lu_id:
                        continue
                    #print 'syn ',synonym, len(self.lu_nodes)
                    if self.lu_nodes.has_key(synonym):
                        #print 'key: ',synonym,' : ',self.lu_nodes[synonym]
                        append=True
                        #for e in new_edges:

                            #if ((e[0] == node and e[1] == self.lu_nodes[synonym]) or (
                            #                e[1] == node and e[0] == self.lu_nodes[synonym])):
                            #    append=False
                        a+=1
                        if append:
                            if (self.lu_nodes[synonym].lu.lu_id,node.lu.lu_id) not in edges and (node.lu.lu_id, self.lu_nodes[synonym].lu.lu_id) not in edges:
                                edges.add((node.lu.lu_id, self.lu_nodes[synonym].lu.lu_id))
                            #new_edges.append((node, self.lu_nodes[synonym]))
                                # for e in lu_graph.all_edges():
                                #    print 'e: ',e
                                #    print e.target()
                                #    if not ((e.target()==node and e.source()==lu_nodes[synonym]) or (e.source()==node and e.target()==lu_nodes[synonym])):
                                #        node, lu_nodes[synonym]
                                # lu_graph.add_edge(node,lu_nodes[synonym],[['rel_id',-8]],True)
                                # f=lu_graph.get_edge(node,lu_nodes[synonym])
                                # print 'F ',f, f.target(), f.source(), f._edge


        #print edges
        print ' xx ', len(edges)
        for edge in edges:
            #print 'edge ',edge
            self.lu_graph.add_edge(self.lu_nodes[edge[0]], self.lu_nodes[edge[1]], [['rel_id', -8]], True)



            # lu_dict=dict()

    MIN = -1000000
    MAX = 3
    MAXS = [3, 3]
    INNER = [True, False]  # ,True,False]
    COUNTER = [500000, 500000]
    node_counter = 0
    inner_synset_rel = True
    node_counter = 100000
    #print 'LU S ', len(lu_synset_dic)
    # find_nearest(node_counter,inner_synset_rel,MAX)
    #finder.find_nearest_simple(lu_graph, list_of_polar, lu_synset_dic, 4, False)  # True)


    #for i in range(0):
     #   MAX = MAXS[i]
     #   inner_synset_rel = False  # INNER[i]
     #   node_counter = COUNTER[i]
     #   finder.find_nearest(lu_graph, list_of_polar, lu_synset_dic, node_counter, inner_synset_rel, MAX)

    #for n in base.all_nodes():
        # c+=1
        # print n, ' syn : ',n.synset.synset_id, ', ',n.synset.lu_set
    #    dic = dict()

        # for nn in n.synset.lu_set:
        #    print '** ',nn.lu_id,' ', nn.lemma,' ', nn.pos,' ',nn.domain,' ', nn.variant
    #    for e in n.all_edges():
    #        if e.target() == n:
    #            s = 0

     #       else:  # source
     #           s = 2
        # xx=1
        #     if dic.has_key(e.rel):
        #         dic[e.rel]=dic[e.rel]+1
        # print ':: ',e.source().synset.synset_id,'---> ', e.target().synset.synset_id,' ',e, ' ',e.rel_id, ' ',e.rel
        # for n2 in e.source().synset.lu_set:
        #    print 'in source: ',n2.lemma,' , ',n2.lu_id
        # for n2 in e.target().synset.lu_set:
        #    print 'in target: ', n2.lemma, ' , ', n2.lu_id
    #    tuple_for_dict = (n.synset.synset_id, dic)
        # if c==10:
        #    break
    #x = set()

    # print x

    # def search_polarization_level(node):
    #    if node.lu_\
path = '/home/aleksandradolega/'

def create_neighbourhood(graph,depth):
    finder=Finder()
    freq_map=finder.find_nearest_simple(graph.lu_graph, graph.list_of_polar, depth=depth)
    return freq_map

def create():

    merged_graph_path=path+'merged_graph.xml.gz'
    lu_graph_path=path+'OUTPUT_GRAPHS_lu.xml.gz'
    gg= GraphReader(lu_graph_path, merged_graph_path, 'localhost', 'root', 'toor', 'wordTEST')#host=None, user=None, passw=None, db_name=None)
    gg.read_graph()
    l=0
    ee=0
    for e in gg.lu_graph.all_edges():
        l+=1
        if e.rel_id==-8:
            ee+=1
    print 'L1: ',l, "( ",ee,")"

    gg.save_graph(path+'withsyn_6_10_15_19_30.xml')

#create()
#c=9/0

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
    n1=0
    n2=0
    for n in node1.all_edges():
        n1+=1
    for n in node2.all_edges():
        n2+=1
    return n1 > n2

print 'GR'
g2=GraphReader(path+'withsyn_6_10_15_19_30.xml',host='localhost',user='root',passw='toor',db_name='wordTEST')
depth=2

pr=Propagator(0,g2.list_of_polar)

X_train=list()
Y_train=list()
for pol in g2.list_of_polar.keys():
    vec,label=pr.get_vector(g2.lu_nodes[pol])
    vec=numpy.asarray(vec)
    #vec=vec.reshape(-1, 1)
    print 'VC ',vec.ndim
    X_train.append(vec)
    Y_train.append(label)

per=0.9
X=[X_train[:int(per*len(X_train))],X_train[int(per*len(X_train)):]]
Y=[Y_train[:int(per*len(Y_train))],Y_train[int(per*len(Y_train)):]]
#neu=NeuralNet()
#neu.create_neural(X[0],Y[0],X[1],Y[1])
old_keys=copy.deepcopy(g2.list_of_polar)
counter=30
good_res=True
while counter>0 and good_res:
    counter-=1
    freq_map=create_neighbourhood(g2,depth)
    print 'freq map ',freq_map
    freq_set = list()
    print 'PR'

    for i in range(1,depth,1):
        freq_set.append(list())
    for k in freq_map.keys():
        if freq_map[k]==0:
            continue
        freq_set[freq_map[k]-1].append(k)
    for i in range(len(freq_set)):
        freq_set[i] = sorted(freq_set[i], cmp=make_comparator(cmpValue),reverse=True)
#for l in sortedDict:
#    print l
#    x=0
#    for f in l.all_edges():
#        x+=1
#    print '>',x
#print 'ss ',sortedDict
    good_res=False
    for i in range(len(freq_set)):
        for elem in freq_set[i]:
            res=pr.evaluate_node_percent(elem)
            if res!=-2:
                good_res=True
print 'END COUNTER ',counter

if True:
    target = open(path+'alldata_30levels_newEdges_onlyNew.txt', 'w')
    #target2 = open(path+'onlynew2.txt', 'w')
    for key in pr.data_dic.keys():
        if key in old_keys.keys():
            continue
        target.write(str(key))
        target.write(', ')
        target.write(g2.lu_nodes[key].lu.lemma)
        target.write(', ')
        target.write(str(g2.lu_nodes[key].lu.variant))
        target.write(', ')
        target.write(str(pr.data_dic[key]))

        #if freq_map.has_key(key):
        #    target.write(', | ')
        #    target.write(str(freq_map[key]))

        target.write('\n')
        #if key not in g2.list_of_polar.keys():
        #    target2.write(key)
        #    target2.write(', ')
        #    target2.write(g2.lu_nodes[key].lu.lemma)
        #    target2.write(', ')
        #    target2.write(str(g2.lu_nodes[key].lu.variant))
        #    target2.write(', ')
        #    target2.write(str(pr.data_dic[key]))
        #    if freq_map.has_key(key):
        #        target2.write(', ')
        #        target2.write(str(freq_map[key]))
        #    target2.write('\n')
    target.close()
#target2.close()


c=0
#for n in g2.lu_nodes():
#for id in g2.list_of_polar:
#    n=g2.lu_nodes[id]
#    c+=1
    #if c==20:
    #    break
#    print 'EV',n
#    pr.evaluate_node_percent(n)
#l=0
#for e in g2.lu_graph.all_edges():
#    l+=1
#print 'L ',l
#for n in g2.lu_graph.all_nodes():

#    for e in n.all_edges():
#        if e.rel_id==-8:
#            print 'N: ',n,' ',e.rel_id,' ',e.source().lu.lemma, e.source().lu.variant, ' -> ', e.target().lu.lemma,e.target().lu.variant
#    print '***'

#g2.save_graph(path+'withsyn_ADDED.xml')