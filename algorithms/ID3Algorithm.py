import random
from anytree import Node, RenderTree

from algorithms.auxiliaryFunctions import *


class ID3Algorithm():

    tst_labels = None

    def __init__(self, F):
        self.F = F

    def fit(self, data, labels):
        all_nodes_finished = False

        tree = dict()  # Tree as a dictionary of dictionaries (dicitionaries is a list of used nodes and indices)
        last_added = []  # Last added nodes

        # Hacer dos bucles, uno por niveles (vertical) y otro por horizontal. Usar last_added

        # FIRST NODE -----------------------------------------------------------------------
        # Compute features random for the particular node
        vec_random = []
        possible_feat = [it for it in range(data.shape[1])]
        for it in range(self.F):
            index_randnum = random.randrange(0, len(possible_feat), 1)
            vec_random.append(possible_feat[index_randnum])
            possible_feat.pop(index_randnum)

        # Find best feature
        information = []
        for feature_number in vec_random:
            information.append(compute_information(data[:,feature_number], labels))
        best_indx = information.index(min(information))
        best_attr = vec_random[best_indx]
        keyy = '['+str(best_attr)+']'
        dictionary = [keyy,[it for it in range(data.shape[0])],'']
        tree[dictionary[0]] = dictionary
        last_added.append(dictionary)

        # REST OF THE NODES ----------------------------------------------------------------
        while len(last_added) > 0:
            new_added = []

            for last_add in last_added:

                # Divide leaf with its children
                att_values, att = number_differentvalues_att(last_add, data)
                Natt_value = len(att_values)
                dict_inst_satis = instances_satisfying_attvalue(data[last_add[1],att], last_add[1])

                # For each children
                for i in range(Natt_value):

                    # Find if the new node is terminal or not
                    indices = dict_inst_satis[att_values[i]]
                    bol_terminal, lab_class = is_terminal(last_add[0], data[indices,:], labels[indices])
                    if bol_terminal == False:
                        # Compute features random for the particular node (that have not been used and that are differ.)
                        vec_random = []
                        used_feat = compute_used_features(last_add[0])
                        possible_feat = [it for it in range(data.shape[1]) if (it not in used_feat)]
                        for it in range(self.F):
                            if len(possible_feat) > 0:
                                index_randnum = random.randrange(0, len(possible_feat), 1)
                                vec_random.append(possible_feat[index_randnum])
                                possible_feat.pop(index_randnum)

                        # Find best feature
                        information = []
                        for feature_number in vec_random:
                            information = information + [compute_information(data[indices,feature_number],
                                                                                 labels[indices]),]
                        best_indx = information.index(min(information))
                        best_attr = vec_random[best_indx]
                        keyy = '[' + str(best_attr) + ']'
                        dictionary = [last_add[0]+keyy, indices, tree[last_add[0]][2]+'*'+str(att_values[i])]

                        # Add node to analyze
                        new_added.append(dictionary)
                    else:
                        sequence = tree[last_add[0]][2]+'*'+str(att_values[i])
                        dictionary = [last_add[0]+'--Class:'+str(lab_class)+'--Seq:'+sequence, dict_inst_satis[
                            att_values[i]],
                                      sequence]

                    # Add node to the tree
                    tree[dictionary[0]] = dictionary

            last_added = new_added

    def classify(self, data_test):
        pass