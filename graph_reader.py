from wosedon.basegraph import BaseGraph
import MySQLdb
from summarizer import Finder
import time


class GraphReader(object):
    finder = Finder()
    lu_synset_dic = dict()
    synsets = list()
    not_disamb_list = list()
    positive_list = list()
    negative_list = list()
    positive_list_strong=list()
    negative_list_strong=list()
    amb_list = list()
    list_of_dicts = list()
    list_of_polar = dict()
    synsets_polar = dict()
    all_synset_relations=set()
    all_lu_relations=set()

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
            self.base.unpickle(self.merged_graph_path)


        self.lu_graph = BaseGraph()
        t=time.time()
        self.lu_graph.unpickle(self.lu_graph_path)
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
        #is_graph_full=False
        if not is_graph_full:
            cur.execute("SELECT l.ID from lexicalunit l where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %') and  (l.comment like '%+ m' or l.comment like '%+ s' or l.comment like '%+ m %' or l.comment like '%+ s %')")

                #"SELECT l.ID from lexicalunit l join lexicalunit l2 on l.lemma=l2.lemma where (l.comment like '%- m' or l.comment like '%- s' or l.comment like '%- m %' or l.comment like '%- s %') and  (l2.comment like '%+ m' or l2.comment like '%+ s' or l2.comment like '%+ m %' or l2.comment like '%+ s %')")

            for row in cur.fetchall():
                self.not_disamb_list.append(row[0])

            cur.execute(
                "SELECT l.ID from lexicalunit l where (l.comment like '%- s' or l.comment like '%- s %')")

            for row in cur.fetchall():
                self.negative_list.append(row[0])

            cur.execute(
                "SELECT l.ID from lexicalunit l where (l.comment like '%- m' or l.comment like '%- m %')")

            for row in cur.fetchall():
                self.negative_list_strong.append(row[0])

            cur.execute(
                "SELECT l.ID from lexicalunit l where (l.comment like '%+ s' or l.comment like '%+ s %')")

            for row in cur.fetchall():
                self.positive_list.append(row[0])

            cur.execute(
                "SELECT l.ID from lexicalunit l where (l.comment like '%+ m' or l.comment like '%+ m %'")

            for row in cur.fetchall():
                self.positive_list_strong.append(row[0])

            cur.execute(
                "SELECT l.ID from lexicalunit l where l.comment like '% amb %' or l.comment like '%A_:0' or l.comment like '%A_: 0'")

            for row in cur.fetchall():
                self.amb_list.append(row[0])

            #print '> ',len(self.amb_list),' ',len(self.negative_list),' ',len(self.positive_list)

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
                print 'po ',n.lu.lu_id,' - ',n.lu.polarity
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
                elif idL in self.positive_list_strong:

                    self.list_of_polar[n.lu.lu_id] = 10
                    n.lu.polarity=10

                elif idL in self.negative_list:

                    self.list_of_polar[n.lu.lu_id] = -1
                    n.lu.polarity=-1
                elif idL in self.negative_list_strong:

                    self.list_of_polar[n.lu.lu_id] = -10
                    n.lu.polarity=-10

                elif idL in self.amb_list:

                    self.list_of_polar[n.lu.lu_id] = 0
                    n.lu.polarity=0
                else:
                    n.lu.polarity = None


    def create_lu_syn_polar_list(self, percent=0.5):

        for n in self.base.all_nodes():
            non = False
            local_polar = list()


            for lu in n.synset.lu_set:


                if not self.list_of_polar.has_key(lu.lu_id):
                    idL = lu.lu_id

                    if idL in self.not_disamb_list:
                        local_polar.append(0)
                        continue
                    if idL in self.positive_list:
                        local_polar.append(1)
                        self.list_of_polar[lu.lu_id] = 1
                    elif idL in self.negative_list_strong:
                        local_polar.append(-10)
                        self.list_of_polar[lu.lu_id] = -10
                    elif idL in self.positive_list_strong:
                        local_polar.append(10)
                        self.list_of_polar[lu.lu_id] = 10
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


        for edge in edges:
            self.lu_graph.add_edge(self.lu_nodes[edge[0]], self.lu_nodes[edge[1]], [['rel_id', -8]], True)
