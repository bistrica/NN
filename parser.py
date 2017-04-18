from graph_reader import GraphReader
from propagator import Propagator

class Parser(object):
    PROPAGATION_TYPE='MANUAL'
    TYPE_MANUAL='MANUAL'
    TYPE_NEURAL='NEURAL'
    MANUAL_RELATION_TYPES=[]
    MANUAL_RELATION_WIGHTS=[]
    LAYERS_UNITS=[]
    LU_PATH=''#
    SYN_PATH=''#
    NEURAL_NETWORK_MODEL_PATH = ''
    SAVE_MERGED_GRAPH_PATH=''#
    SAVE_MODIFIED_MERGED_GRAPH_PATH = ''
    HOST=''#
    USER=''#
    PASS=''#
    DB_NAME=''#
    RELS_TO_APPEND=''#
    DEPTH = 0
    TRAINING_DEPTH = 0
    PERCENT = 0

    #parameters=[]
    def __init__(self, config_path):
        f = open(config_path, 'r')
        counter=0
        try:
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
                            line[1]=line[1].replace('[','')
                            line[1]=line[1].replace(']', '')
                            if line[1].strip()=='':
                                line[1]=[]
                            else:
                                line[1].replace(',', ' ')
                                line[1]=list(line[1].split())
                        elif getattr(self,line[0]).isdigit():
                            line[1]=float(line[1])

                        setattr(self, line[0],line[1])
        except:
            raise ValueError('Wrong input in line ',counter,' in config file.')

    def main(self):
        if self.SYN_PATH!='' and self.LU_PATH !='':
            all_rels=self.RELS_TO_APPEND=='all'
            gg = GraphReader(self.LU_PATH, self.SYN_PATH, host=self.HOST, user=self.USER, passw=self.PASS, db_name=self.DB_NAME,
                             all_rels=all_rels, rel_to_add=self.RELS_TO_APPEND)
            gg.read_graph()
            if self.SAVE_MERGED_GRAPH_PATH!='':
                gg.save_graph(self.SAVE_MERGED_GRAPH_PATH)
        elif self.LU_PATH !='':
            gg = GraphReader(self.LU_PATH, host=self.HOST, user=self.USER, passw=self.PASS, db_name=self.DB_NAME)
            if self.MANUAL_RELATION_TYPES == 'all':
                self.MANUAL_RELATION_TYPES = gg.get_all_relations()
            if self.PROPAGATION_TYPE==self.TYPE_MANUAL:
                pr = Propagator(Propagator.MANUAL, gg.list_of_polar, percent=self.PERCENT, depth=self.DEPTH, rel_ids=self.MANUAL_RELATION_TYPES,weights=self.MANUAL_RELATION_WIGHTS)
            elif self.PROPAGATION_TYPE==self.TYPE_NEURAL:
                pr = Propagator(Propagator.NEURAL, gg.list_of_polar, percent=self.PERCENT, depth=self.DEPTH, rel_ids=self.MANUAL_RELATION_TYPES)



p=Parser('/home/aleksandradolega/config')
p.main()