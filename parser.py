

class Parser(object):

    parameters=[]
    def __init__(self, config_path):
        f = open(config_path, 'r')
        for line in f:
            print line
            if '#' in line:
                continue
            else:
                line=line.split(':')
                print line
                if len(line)==2:
                    line[1]=line[1].replace('\n','')
                    line[1] = line[1].strip()
                    self.parameters.append((line[0],line[1]))
        print self.parameters


p=Parser('/home/aleksandradolega/config')