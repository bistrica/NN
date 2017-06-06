from os.path import isfile,join
from os import listdir

def summarize():
    dir='/home/aleksandradolega/PODSUMA/'

    path1='bayes_75_depth_2_training_1.txt'#NEURAL_5layers_75per_norm_allrels_sortedByVal_DEPTH_1.txt'#'NEURAL_MULTIPLE_3layers_50per_norm_allrels_sortedByVal_DEPTH_1.txt'
    path2='svm_75_depth_2_training_1.txt'#SVM_30per_norm_allrels_sortedByVal.txt'#''NEURAL_[256,64,128,32,64,16]layers_50per_norm_allrels_sortedByVal_DEPTH_1.txt'#'NEURAL_4layers_75per_norm_allrels_sortedByVal_DEPTH_1.txt'
    save=dir+'POROWNANIE_'+path1+'_'+path2
    path1=dir+path1
    path2=dir+path2
    paths = [path1, path2]

    files=[]
    files_dics=[]
    for p in paths:
        files.append(open(p, 'r'))
        files_dics.append(dict())
    save_file=open(save,'w+')
    #f1 = open(path1, 'r')
    #f2=open(path2,'r')

    try:
        for i in range(len(files)):
            f1=files[i]
            for line in f1:
                line=line.replace('\n','')
                id = line.split(',')
                files_dics[i][id[0]]=line
    except:
        print 'Reading failed!'

    #for k in dic1.keys():
    #    if k not in dic2.keys():
    #        print dic1[k]
    #print '****'
    #for k in dic2.keys():
    #    if k not in dic1.keys():
    #        print dic2[k]
    #print '*******'
    C=0
    dic2=files_dics[0]
    for k in dic2.keys():
        for i in range (1,len(files_dics)):
            dic1=files_dics[i]
            if k in dic1.keys() and dic1[k]!=dic2[k]:
                C+=1
                print k,'; ',files[i],':: ',dic1[k]+' '+dic2[k]
        print '====='
            #save_file.write(dic1[k]+' '+dic2[k]+'\n')
    print C
    #save_file.write('RAZEM '+str(C))
    #save_file.close()

def correct():
    dir='/home/aleksandradolega/'
    files=[f for f in listdir(dir) if isfile(join(dir,f)) and 'POROWNANIE' in f]
    for path in files:
        path=dir+path
        summ_file=open(path,'r')
        summ_correct=open(path+'_correct.txt','w+')
        for line in summ_file:
            line=line.replace('] ','}')
            line=line.replace(']',']\n')
            line = line.replace('}', '] ')
            summ_correct.write(line)
        summ_correct.close()

#correct()
summarize()