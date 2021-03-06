from graph_reader import GraphReader
from propagator import Propagator
#import _thread
import time
import copy
import pickle
import time

class Parser(object):
    PROPAGATION_TYPE='MANUAL'
    TYPE_MANUAL='MANUAL'
    TYPE_NEURAL='NEURAL'
    TYPE_NEURAL_MULTIPLE='NEURAL_MULTIPLE'
    TYPE_KNN='KNN'
    TYPE_SVM='SVM'
    TYPE_BAYES='BAYES'
    TYPE_ENSEMBLE='ENSEMBLE'

    SAVE_TO_DATABASE=0
    TYPES=[TYPE_NEURAL,TYPE_MANUAL,TYPE_BAYES,TYPE_NEURAL_MULTIPLE,TYPE_SVM,TYPE_KNN,TYPE_ENSEMBLE]
    MANUAL_RELATION_TYPES=[]
    MANUAL_RELATION_WIGHTS=[]
    LAYERS_UNITS=[]
    LAYERS_UNITS_NM = []
    LU_PATH=''
    MERGED_PATH=''
    NEURAL_NETWORK_MODEL_PATH = ''
    SVM_MODEL_PATH=''
    SAVE_SVM_MODEL_PATH = ''
    SAVE_MERGED_GRAPH_PATH=''
    SAVE_MODIFIED_MERGED_GRAPH_PATH = ''
    SAVE_NEURAL_NETWORK_MODEL_PATH=''
    SAVE_ENSEMBLE_PATH=''
    ENSEMBLE_PATH=''
    HOST=''
    USER=''
    PASS=''
    DB_NAME=''
    RELS_TO_APPEND=''
    DEPTH = 0
    TRAINING_DEPTH = 0
    PERCENT = 0
    CHOSEN_POS=[]
    SVM_KERNEL = None
    KNN_NEIGHBOURS_NUMBER=None
    KNN_WEIGHTS=None
    KNN_ALGORITHM=None
    NORMALIZATION=0
    CLASSIFY_DATA=[]
    FILE_LEX_UNITS_WITH_NEW_POLARITY=''


    def __init__(self, config_path):
        f = open(config_path, 'r')
        counter=0
        try:
        #if True:
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
                                        line[1][i]=float(line[1][i])
                        elif isinstance(getattr(self,line[0]),int):
                            line[1]=float(line[1])
                        if line[1]=='':
                            line[1]=None
                        if isinstance(line[1],list) and len(line[1])==0:
                            line[1]=None
                        setattr(self, line[0],line[1])
        except:
            raise ValueError('Wrong input in line '+str(counter)+' in config file.')
        if self.NORMALIZATION==0:
            self.NORMALIZATION=False
        else:
            self.NORMALIZATION=True

    def main(self):
        if self.MERGED_PATH is not None and self.LU_PATH is not None:
            all_rels=self.RELS_TO_APPEND=='all'
            gg = GraphReader(self.LU_PATH, self.MERGED_PATH, host=self.HOST, user=self.USER, passw=self.PASS, db_name=self.DB_NAME,
                             all_rels=all_rels, rel_to_add=self.RELS_TO_APPEND)
            gg.read_graph()
            if self.SAVE_MERGED_GRAPH_PATH is not None and self.SAVE_MERGED_GRAPH_PATH!='':
                gg.save_graph(self.SAVE_MERGED_GRAPH_PATH)

            if self.PROPAGATION_TYPE in self.TYPES:
                self.choose_propagation(gg)

        elif self.LU_PATH is not None and self.PROPAGATION_TYPE in self.TYPES:
            gg = GraphReader(self.LU_PATH, host=self.HOST, user=self.USER, passw=self.PASS, db_name=self.DB_NAME)
            self.choose_propagation(gg)

    def choose_propagation(self, graph):
        pr=None
        self.SAVE_TO_DATABASE = self.SAVE_TO_DATABASE==1
        if self.MANUAL_RELATION_TYPES == 'all':
            self.MANUAL_RELATION_TYPES = graph.get_all_relations()
        if self.PROPAGATION_TYPE == self.TYPE_MANUAL:
            pr = Propagator(type=Propagator.MANUAL, known_data_dic=graph.list_of_polar, graph=graph, percent=self.PERCENT, depth=self.DEPTH, normalization=self.NORMALIZATION,
                            rel_ids=self.MANUAL_RELATION_TYPES, weights=self.MANUAL_RELATION_WIGHTS, save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY,save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)
        elif self.PROPAGATION_TYPE == self.TYPE_NEURAL:
            pr = Propagator(type=Propagator.NEURAL, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH, training_depth=self.TRAINING_DEPTH, normalization=self.NORMALIZATION,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES, neural_layers=self.LAYERS_UNITS,
                            network=self.NEURAL_NETWORK_MODEL_PATH, save_network=self.SAVE_NEURAL_NETWORK_MODEL_PATH, save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY,save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)
        elif self.PROPAGATION_TYPE == self.TYPE_NEURAL_MULTIPLE:
            pr = Propagator(type=Propagator.NEURAL_MULTIPLE, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH, normalization=self.NORMALIZATION,
                            training_depth=self.TRAINING_DEPTH,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES, neural_layers_multiple=self.LAYERS_UNITS_NM,
                            network=self.NEURAL_NETWORK_MODEL_PATH, save_network=self.SAVE_NEURAL_NETWORK_MODEL_PATH,
                            save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY,chosen_pos=self.CHOSEN_POS,save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)

        elif self.PROPAGATION_TYPE == self.TYPE_BAYES:
            pr = Propagator(type=Propagator.BAYES, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH, normalization=self.NORMALIZATION,
                            training_depth=self.TRAINING_DEPTH,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES, neural_layers=self.LAYERS_UNITS,
                            save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY,save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)
        elif self.PROPAGATION_TYPE == self.TYPE_KNN:
            pr = Propagator(type=Propagator.KNN, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH, normalization=self.NORMALIZATION,
                            training_depth=self.TRAINING_DEPTH,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES, neural_layers=self.LAYERS_UNITS, knn_algorithm=self.KNN_ALGORITHM, knn_weights=self.KNN_WEIGHTS, neighbours_number=self.KNN_NEIGHBOURS_NUMBER,
                            save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY,save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)
        elif self.PROPAGATION_TYPE == self.TYPE_SVM:
            pr = Propagator(type=Propagator.SVM, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH,normalization=self.NORMALIZATION,
                            training_depth=self.TRAINING_DEPTH,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES,
                            save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY, kernel=self.SVM_KERNEL,svm_model=self.SVM_MODEL_PATH,save_svm_path=self.SAVE_SVM_MODEL_PATH,save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)
        elif self.PROPAGATION_TYPE==self.TYPE_ENSEMBLE:
            pr = Propagator(type=Propagator.ENSEMBLE, known_data_dic=graph.list_of_polar, graph=graph, depth=self.DEPTH,
                            normalization=self.NORMALIZATION,
                            training_depth=self.TRAINING_DEPTH,
                            percent=self.PERCENT, rel_ids=self.MANUAL_RELATION_TYPES,
                            save_new_lu_polarities=self.FILE_LEX_UNITS_WITH_NEW_POLARITY, kernel=self.SVM_KERNEL,chosen_pos=self.CHOSEN_POS, neural_layers=self.LAYERS_UNITS, neural_layers_multiple=self.LAYERS_UNITS_NM, ensemble_path=self.ENSEMBLE_PATH, save_ensemble_path=self.SAVE_ENSEMBLE_PATH, save_to_db=self.SAVE_TO_DATABASE,only_classify_list=self.CLASSIFY_DATA)

        if pr is not None:
            t=time.time()
            pr.propagate()
            t = time.time()-t

        if self.SAVE_MODIFIED_MERGED_GRAPH_PATH is not None:
            graph.save_graph(self.SAVE_MODIFIED_MERGED_GRAPH_PATH)

p=Parser('/home/aleksandradolega/config')
p.main()