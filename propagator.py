from wosedon.basegraph import BaseGraph

class Propagator(object):


    PERCENT=0.5
    REL_IDS=[]
    WEIGHTS=[]
    MANUAL = 0
    NEURAL_NETWORK=1

    rel_positive=dict()
    rel_negative=dict()
    rel_none=dict()
    rel_amb=dict()
    data_dic=None
    C=0
    CN=0
    #[-8, 10, 11, 12, 62, 104, 141, 169, 244]
    #-8:synonimia, 12-antonimia, 11-hiponimia,10-hiperonimia, 62-syn.miedzyparadygmatyczna,104-antonimia wlasciwa,141-syn.miedzypar.,169-syn.mmiedzy,244-syn..miedzypar
    def __init__(self, type, known_data_dic, rel_ids=[-8,10,11,12,62,104,141,169,244], weights=[10,2,1,-10,10,-10,10,10,10]):
        self.data_dic=known_data_dic
        for k in known_data_dic.keys():
            print k,' ',known_data_dic[k]
        self.REL_IDS=rel_ids
        self.WEIGHTS=weights



    def simple_propagate(self):
        x=0

    def evaluate_node_percent(self,node,percent=0.5):
        self.CN+=1
        self.rel_positive = dict()
        self.rel_negative = dict()
        self.rel_none = dict()
        self.rel_amb = dict()
        count=0
        none=0
        for e in node.all_edges():

            #print 'r: ',e.rel_id
            if e.rel_id in self.REL_IDS:
                count+=1
               # print 'REL IDS',self.REL_IDS
                polarity=0
                scnd_node=None
                if e.source()==node:
                    scnd_node=e.target()

                else:
                    scnd_node=e.source()
                dic_to_update=dict()
                #print ':: ',scnd_node.lu.lu_id, self.data_dic.has_key(scnd_node.lu.lu_id)
                if self.data_dic.has_key(scnd_node.lu.lu_id):
                    polarity=self.data_dic[scnd_node.lu.lu_id]

                    if polarity>0:
                        dic_to_update = self.rel_positive
                        #print 'p: ',len(self.rel_positive)
                    elif polarity<0:
                        dic_to_update = self.rel_negative
                        #print 'n: ', len(self.rel_negative)
                    else:
                        dic_to_update = self.rel_amb
                        #print 'a: ', len(self.rel_amb)

                else:
                    none+=1
                    dic_to_update=self.rel_none
                    #print 'on: ', len(self.rel_none)
                if dic_to_update.has_key(e.rel_id):
                    dic_to_update[e.rel_id]+=1
                else:
                    dic_to_update[e.rel_id] = 1
        vector_p=list()
        vector_n = list()
        vector_a = list()
        vec_tuples=list()

        if none<=percent*count:
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

            print 'Pos: ', pos, ' neg: ', neg, ', amb ', amb, "(", node.lu.lemma, ',', node.lu.variant, ' - ', \
            self.data_dic[node.lu.lu_id]

            if pos>neg and pos>amb:
                res=1
            elif neg>pos and neg>amb:
                res=-1
            elif amb>pos and amb>neg:
                res=0
            else:
                res=-2
            if res==self.data_dic[node.lu.lu_id]:
                self.C+=1
            print 'C/CN ',self.C,' ',self.CN
            #if pos > neg and pos > amb

                #if self.rel_none.has_key(rel):
                #    vector.append(self.rel_none[rel])
                #else:
                #    vector.append(0)
        #print 'vec ',vector
        #for i in range(len(vector)):
        #    if vector[i] is None:
        #        vector[i]=0
        #print 'vec2 ',vector






