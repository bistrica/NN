from graph_reader import GraphReader
from propagator import Propagator

class Parser(object):
    PROPAGATION_TYPE='MANUAL'#
    TYPE_MANUAL='MANUAL'
    TYPE_NEURAL='NEURAL'
    TYPES=[TYPE_NEURAL,TYPE_MANUAL]
    MANUAL_RELATION_TYPES=[]#
    MANUAL_RELATION_WIGHTS=[]#
    LAYERS_UNITS=[]#
    LU_PATH=''#
    MERGED_PATH=''#
    NEURAL_NETWORK_MODEL_PATH = ''#
    SAVE_MERGED_GRAPH_PATH=''#
    SAVE_MODIFIED_MERGED_GRAPH_PATH = ''#
    SAVE_NEURAL_NETWORK_MODEL_PATH=''
    HOST=''#
    USER=''#
    PASS=''#
    DB_NAME=''#
    RELS_TO_APPEND=''#
    DEPTH = 0#
    TRAINING_DEPTH = 0#
    PERCENT = 0#
    FILE_LEX_UNITS_WITH_NEW_POLARITY=''

    #parameters=[]
    def __init__(self, config_path):
        f = open(config_path, 'r')
        counter=0
        #try:
        if True:
            for line in f:
                counter+=1
                if line.strip()=='':
                    continue
                if '#' in line:
                    continue
                else:
                    line=line.split(':')
                    #print line
                    if len(line)==2:

                        line[1]=line[1].replace('\n','')
                        line[1] = line[1].strip()
                        if isinstance(getattr(self,line[0]),list):
                            if '[' in line[1]:

                                line[1]=line[1].replace('[','')
                                line[1]=line[1].replace(']', '')
                                if line[1].strip()=='':
                                    line[1]=[]
                                else:
                                    line[1]=line[1].replace(',', ' ')
                                    line[1]=list(line[1].split())
                                    for i in range(len(line[1])):
                                        line[1][i]=int(line[1][i])
                        elif isinstance(getattr(self,line[0]),int):
                            print 'digit',line[0],line[1]
                            line[1]=float(line[1])

                        setattr(self, line[0],line[1])
        #except:
        #    raise ValueError('Wrong input in line '+str(counter)+' in config file.')

    def main(self):
        if self.MERGED_PATH!='' and self.LU_PATH !='':
            all_rels=self.RELS_TO_APPEND=='all'
            gg = GraphReader(self.LU_PATH, self.MERGED_PATH, host=self.HOST, user=self.USER, passw=self.PASS, db_name=self.DB_NAME,
                             all_rels=all_rels, rel_to_add=self.RELS_TO_APPEND)
            gg.read_graph()
            if self.SAVE_MERGED_GRAPH_PATH!='':
                gg.save_graph(self.SAVE_MERGED_GRAPH_PATH)

            if self.PROPAGATION_TYPE in self.TYPES:
                self.choose_propagation(gg)

        elif self.LU_PATH !='' and self.PROPAGATION_TYPE in self.TYPES:
            gg = GraphReader(self.LU_PATH, host=self.HOST, user=self.USER, passw=self.PASS, db_name=self.DB_NAME)
            self.choose_propagation(gg)

    def choose_propagation(self, graph):
        if self.MANUAL_RELATION_TYPES == 'all':
            self.MANUAL_RELATION_TYPES = graph.get_all_relations()
        if self.PROPAGATION_TYPE == self.TYPE_MANUAL:
            pr = Propagator(type=Propagator.MANUAL, known_data_dic=graph.list_of_polar, graph=graph, percent=self.PERCENT, depth=self.DEPTH,
                            rel_ids=self.MANUAL_RELATION_TYPES, weights=self.MANUAL_RELATION_WIGHTS, save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY)
        elif self.PROPAGATION_TYPE == self.TYPE_NEURAL:
            pr = Propagator(type=Propagator.NEURAL, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH, training_depth=self.TRAINING_DEPTH,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES, neural_layers=self.LAYERS_UNITS,
                            network=self.NEURAL_NETWORK_MODEL_PATH, save_network=self.SAVE_NEURAL_NETWORK_MODEL_PATH, save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY)
        pr.propagate()

        if self.SAVE_MODIFIED_MERGED_GRAPH_PATH != '':
            graph.save_graph(self.SAVE_MODIFIED_MERGED_GRAPH_PATH)
p=Parser('/home/aleksandradolega/config')
p.main()