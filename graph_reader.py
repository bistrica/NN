
from wosedon.basegraph import BaseGraph

import MySQLdb
from sklearn.neural_network import MLPClassifier
from summarizer import Finder
from propagator import Propagator
from Neural import Neural
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
    all_synset_relations=set()
    all_lu_relations=set()
    #path = '/home/aleksandradolega/'
    lu_graph_path = ''
    merged_graph_path = ''

    polar_nodes = list()
    synsets_polar = dict()
    lu_nodes = dict()
    lu_graph=BaseGraph
    base = BaseGraph()

    def __init__(self, lu_graph_path, merged_graph_path=None, host=None, user=None, passw=None, db_name=None, all_rels=False, rel_to_add=[10,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,29,30]):
        self.added_relations=rel_to_add

        self.lu_graph_path = lu_graph_path
        full_lu_graph=False#merged_graph_path is None


        if merged_graph_path is not None:


            self.merged_graph_path = merged_graph_path

            self.base = BaseGraph()
            self.base.unpickle(self.merged_graph_path)  # path + 'merged_graph.xml.gz') path = '/home/aleksandradolega/'


        self.lu_graph = BaseGraph()
        t=time.time()
        self.lu_graph.unpickle(self.lu_graph_path)  # path + 'OUTPUT_GRAPHS_lu.xml.gz')
        for n in self.lu_graph.all_nodes():
            if hasattr(n.lu,'polarity'):
                full_lu_graph=True
            break

        if host is not None:
            t=time.time()
            self.get_data_from_db(host, user, passw, db_name, all_rels, full_lu_graph)
            t = time.time() - t
            print 'Time0 ', t
            t = time.time()

        for e in self.lu_graph.all_edges():
            self.all_lu_relations.add(e.rel_id)
        t=time.time()-t
        print 'Time1 ',t
        t = time.time()
        if True:
            self.create_map_lu_node()
            t = time.time() - t
            print 'Time2 ', t
            t = time.time()
            self.create_lu_polar_list()
            t = time.time() - t
            print 'Time3 ', t


    def get_all_relations(self):
        rels=list(self.all_synset_relations)
        rels.extend(list(self.all_lu_relations))
        return rels


    def get_data_from_db(self, host, user, passw, db_name, all_rels, is_graph_full):
        db = MySQLdb.connect(host=host,  # "localhost",    # your host, usually localhost
                             user=user,  # "root",         # your username
                             passwd=passw,  # "toor",  # your password
                             db=db_name)  # "wordTEST")        # name of the data base

        cur = db.cursor()
        if not is_graph_full:
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

        cur.execute(
            "SELECT REL_ID from synsetrelation")

        for row in cur.fetchall():
            self.all_synset_relations.add(row[0])
        if all_rels:
            self.added_relations=list(self.all_synset_relations)


        db.close()



    def print_pos_neg(self, synsets_polar):

        pos = list()
        neg = list()

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


    def create_map_lu_node(self):
        for node in self.lu_graph.all_nodes():
            self.lu_nodes[node.lu.lu_id] = node


    def create_lu_polar_list(self):

        for n in self.lu_graph.all_nodes():
            if hasattr(n.lu, 'polarity') and n.lu.polarity!=None:
                self.list_of_polar[n.lu.lu_id] = n.lu.polarity
                continue
            if True:
                idL = n.lu.lu_id


                if idL in self.not_disamb_list:

                    n.lu.polarity = None
                    continue
                elif idL in self.positive_list:

                    self.list_of_polar[n.lu.lu_id] = 1
                    n.lu.polarity=1

                elif idL in self.negative_list:

                    self.list_of_polar[n.lu.lu_id] = -1
                    n.lu.polarity=-1

                elif idL in self.amb_list:

                    self.list_of_polar[n.lu.lu_id] = 0
                    n.lu.polarity=0
                else:
                    n.lu.polarity = None


    def create_lu_syn_polar_list(self, percent=0.5):

        for n in self.base.all_nodes():
            non = False
            local_polar = list()
            # synsets.append(n.synset)

            for lu in n.synset.lu_set:

                # print '> ',lu.lu_id
                # if not lu_synset_dic.has_key(lu.lu_id):
                #    lu_synset_dic[lu.lu_id]=n.synset#.lu_id]=n.synset

                if not self.list_of_polar.has_key(lu.lu_id):
                    idL = lu.lu_id

                    if idL in self.not_disamb_list:
                        local_polar.append(0)
                        continue
                    if idL in self.positive_list:
                        local_polar.append(1)
                        self.list_of_polar[lu.lu_id] = 1

                    elif idL in self.negative_list:
                        local_polar.append(-1)
                        self.list_of_polar[lu.lu_id] = -1

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

                            edges.append((self.lu_nodes[source_lu.lu_id],self.lu_nodes[target_lu.lu_id],synset_edge.rel_id))


        for edge in edges:
            self.lu_graph.add_edge(edge[0], edge[1], [['rel_id', edge[2]]], True)

    def read_graph(self):
        self.create_lu_syn_polar_list()
        self.create_map_lu_node()

        self.append_synonymy_edges()
        print 'APPENDING RELATIONS'
        self.append_synset_relations()
        print 'STOP APPENDING RELATIONS'


    def save_graph(self, path):
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

                    if self.lu_nodes.has_key(synonym):

                        append=True

                        a+=1
                        if append:
                            if (self.lu_nodes[synonym].lu.lu_id,node.lu.lu_id) not in edges and (node.lu.lu_id, self.lu_nodes[synonym].lu.lu_id) not in edges:
                                edges.add((node.lu.lu_id, self.lu_nodes[synonym].lu.lu_id))

        print ' xx ', len(edges)
        for edge in edges:
            self.lu_graph.add_edge(self.lu_nodes[edge[0]], self.lu_nodes[edge[1]], [['rel_id', -8]], True)

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




#create()
#c=9/0





print 'END COUNTER ',counter

if False:
    target = open(path+'alldata_30levels_allEdges_onlyNew.txt', 'w')
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