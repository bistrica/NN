
from wosedon.basegraph import BaseGraph

import MySQLdb
from sklearn.neural_network import MLPClassifier
from summarizer import Finder


class GraphReader(object):
    finder = Finder()
    lu_synset_dic = dict()
    synsets = list()
    not_disamb_list = list()
    positive_list = list()  # cur.fetchall()
    negative_list = list()  # cur.fetchall()
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

    def __init__(self, lu_graph_path, merged_graph_path=None, host=None, user=None, passw=None, db_name=None):
        if host is not None:
            self.get_data_from_db(host, user, passw, db_name)
        if merged_graph_path is not None:


            self.merged_graph_path = merged_graph_path
            self.lu_graph_path = lu_graph_path
            self.base = BaseGraph()
            self.base.unpickle(self.merged_graph_path)  # path + 'merged_graph.xml.gz') path = '/home/aleksandradolega/'


        self.lu_graph = BaseGraph()
        self.lu_graph.unpickle(self.lu_graph_path)  # path + 'OUTPUT_GRAPHS_lu.xml.gz')
        if merged_graph_path is None:
            self.create_lu_syn_polar_list()

    #def __init__(self, final_graph_path):
    #    self.lu_graph = BaseGraph()
    #    self.lu_graph.unpickle(final_graph_path)

    def get_data_from_db(self, host, user, passw, db_name):
        db = MySQLdb.connect(host=host,  # "localhost",    # your host, usually localhost
                             user=user,  # "root",         # your username
                             passwd=passw,  # "toor",  # your password
                             db=db_name)  # "wordTEST")        # name of the data base

        cur = db.cursor()

        cur.execute(
            "SELECT l.ID from lexicalunit l join lexicalunit l2 on l.lemma=l2.lemma where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %') and  (l2.comment like '%+ m' or l2.comment like '%+ s' or l2.comment like '%+ m %' or l2.comment like '%+ s %')")

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

    def create_lu_polar_list(self, percent):

        for lu in self.lu_graph:
            if not self.list_of_polar.has_key(lu.lu_id):
                idL = lu.lu_id  # str(lu.lu_id)+"L"


                if idL in self.not_disamb_list:
                    self.list_of_polar[lu.lu_id] = 0
                    continue
                elif idL in self.positive_list:

                    self.list_of_polar[lu.lu_id] = 1
                    # print 'POS'
                elif idL in self.negative_list:

                    self.list_of_polar[lu.lu_id] = -1
                    # print 'NEG'
                else:
                    self.list_of_polar[lu.lu_id] = -2
                    # list_of_polar[lu.lu_id] = 0 #zakom. do testu rozpiecia



    def create_lu_syn_polar_list(self, percent):
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




    def read_graph(self):
        self.create_lu_syn_polar_list()
        self.create_map_lu_node()
        self.append_synonymy_edges()


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
        for node in self.lu_graph.all_nodes():
            if self.lu_synset_dic.has_key(node.lu.lu_id):
                synonyms = self.lu_synset_dic[node.lu.lu_id]

                for synonym in synonyms:

                    if self.lu_nodes.has_key(synonym):
                        for e in new_edges:
                            if not ((e.target() == node and e.source() == self.lu_nodes[synonym]) or (
                                            e.source() == node and e.target() == self.lu_nodes[synonym])):
                                new_edges.append((node, self.lu_nodes[synonym]))
                                # for e in lu_graph.all_edges():
                                #    print 'e: ',e
                                #    print e.target()
                                #    if not ((e.target()==node and e.source()==lu_nodes[synonym]) or (e.source()==node and e.target()==lu_nodes[synonym])):
                                #        node, lu_nodes[synonym]
                                # lu_graph.add_edge(node,lu_nodes[synonym],[['rel_id',-8]],True)
                                # f=lu_graph.get_edge(node,lu_nodes[synonym])
                                # print 'F ',f, f.target(), f.source(), f._edge

        for edge in new_edges:
            self.lu_graph.add_edge(edge[0], edge[1], [['rel_id', -8]], True)


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
    #    if node.lu_

path = '/home/aleksandradolega/'
merged_graph_path=path+'merged_graph.xml.gz'
lu_graph_path=path+'OUTPUT_GRAPHS_lu.xml.gz'
gg= GraphReader(lu_graph_path, merged_graph_path, 'localhost', 'root', 'toor', 'wordTEST')#host=None, user=None, passw=None, db_name=None)
gg.read_graph()
gg.save_graph(path+'withsyn')