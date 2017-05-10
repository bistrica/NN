from os.path import isfile,join
from os import listdir

def summarize():
    dir='/home/aleksandradolega/'
    path1='NEURAL_5layers_75per_norm_allrels_sortedByVal_DEPTH_1.txt'#'NEURAL_MULTIPLE_3layers_50per_norm_allrels_sortedByVal_DEPTH_1.txt'
    path2='NEURAL_4layers_75per_norm_allrels_sortedByVal_DEPTH_1.txt'
    save=dir+'POROWNANIE_'+path1+'_'+path2
    path1=dir+path1
    path2=dir+path2
    save_file=open(save,'w+')
    f1 = open(path1, 'r')
    f2=open(path2,'r')
    lines1=list()
    lines2=list()
    dic1=dict()
    dic2=dict()
    try:

        for line in f1:
            line=line.replace('\n','')
            id = line.split(',')
            dic1[id[0]]=line
        for line in f2:
            line = line.replace('\n', '')
            id = line.split(',')
            dic2[id[0]] = line
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
    for k in dic2.keys():
        if k in dic1.keys() and dic1[k]!=dic2[k]:
            C+=1
            print dic1[k]+' '+dic2[k]
            save_file.write(dic1[k]+' '+dic2[k]+'\n')
    print C
    save_file.write('RAZEM '+str(C))
    save_file.close()

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