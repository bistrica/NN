class Finder(object):
    #print_pos_neg()
    def find_nearest(self,lu_graph, list_of_polar, lu_synset_dic, node_counter,inner_synset_rel,MAX, MIN):
        lu_dict=dict()
        for node in lu_graph.all_nodes():
            #print 'N ', node_counter
            node_counter -= 1
            if node_counter == 0:
                #'COUNTER STOP'
                break

            current = node
            distance = 0
            visited = list()
            queue = list()
            queue_level = dict()
            queue_level[current] = 0
            while current.lu.lu_id not in list_of_polar:
                if queue_level[current] >= MAX:
                    distance = 2 * MIN
                    #print 'BREAK - DEEP'
                    break
                visited.append(current)
                found = False
                if inner_synset_rel:
                    synset=None
                    if lu_synset_dic.has_key(current.lu.lu_id):
                        synset = lu_synset_dic[current.lu.lu_id]
                        for synonym in synset:
                            if synonym in list_of_polar:
                                distance = queue_level[current] + 1
                                found = True
                                break
                    else:
                        print 'WRONG KEY ', current.lu.lu_id,' ',current.lu.lemma,' ',current.lu.variant

                    if found:
                        #print 'SYNONYM'
                        break

                for item in current.all_edges():
                    # print item.source().lu.lu_id," (",item.source().lu.lemma,") --> ",item.target().lu.lu_id," (",item.target().lu.lemma,")",item.rel_id
                    if current == item.source():
                        target = item.target()
                        if target not in visited:
                            queue.append(target)
                            queue_level[target] = queue_level[current] + 1
                    if current == item.target():
                        source = item.source()
                        if source not in visited:
                            queue.append(source)
                            queue_level[source] = queue_level[current] + 1
                if len(queue) == 0:
                    distance = MIN
                    #print 'BREAK'
                    break
                current = queue[0]
                queue.remove(current)
                # if queue_level[current]>MAX:
                #    print 'BREAK - DEEP'
                #    break

            if distance > MIN:
                distance = queue_level[current]
            lu_dict[node.lu.lu_id] = distance
            #print 'LU DIC : ', len(lu_dict)

        print 'LU DIC : ', len(lu_dict)
        frequency_dic = dict()
        for key in lu_dict.keys():
            if frequency_dic.has_key(lu_dict[key]):
                frequency_dic[lu_dict[key]] += 1
            else:
                frequency_dic[lu_dict[key]] = 1
        print 'FREQ ', frequency_dic
        print 'synonimy  ', inner_synset_rel

    def find_nearest_simple(self,lu_graph, list_of_polar, lu_synset_dic=None, depth=5, synset_rel=False):
        distances=dict()
        polarized_nodes=list()
        print 'lis ',len(list_of_polar)
        for node in lu_graph.all_nodes():
            if node.lu.lu_id in list_of_polar:
                polarized_nodes.append(node)
                distances[node]=0
        level=1
        while (level!=depth):
            new_polar_nodes=list()
            for node in polarized_nodes:

                if synset_rel:

                    if lu_synset_dic.has_key(node.lu.lu_id):
                        synset = lu_synset_dic[node.lu.lu_id]
                        for synonym in synset:#.lu_set:
                            if synonym in list_of_polar:
                                if not (node in distances and distances[node]==0):
                                    distances[node] = 1
                                break


                for edge in node.all_edges():
                    if node == edge.source():
                        target = edge.target()
                        if target not in distances:
                            distances[target]=level
                            new_polar_nodes.append(target)
                    if node == edge.target():
                        source = edge.source()
                        if source not in distances:
                            distances[source] = level
                            new_polar_nodes.append(source)

            level+=1
            polarized_nodes=new_polar_nodes
        #print 'DIC: ',distances
        print 'DIC LEN: ', len(distances)
        frequency_dic = dict()
        for key in distances.keys():
            if frequency_dic.has_key(distances[key]):
                frequency_dic[distances[key]] += 1
            else:
                frequency_dic[distances[key]] = 1
        print 'FREQ ', frequency_dic
        return distances