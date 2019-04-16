import math
import numpy as np


def compute_information(data, labels):
    dat = dict()
    N_instances = len(labels)

    for it in range(N_instances):
        dicti = dict()
        item = data[it]
        if item not in dat:
            dicti[labels[it]] = 1
            dat[item] = dicti
        else:
            if labels[it] in dat[item]:
                dat[item][labels[it]] = dat[item][labels[it]] + 1
            else:
                last_dict = dict(dat[item])
                last_dict[labels[it]] = 1
                dat[item] = dict()
                dat[item] = last_dict
    information = 0
    for item in set(data):
        num_firstPart = sum(data==item)
        first_part = num_firstPart/len(data)
        logpart = 0
        for dictkeys in dat[item].keys():
            vall = dat[item][dictkeys]
            if vall != 0:
                coef = vall/num_firstPart
                logpart = logpart + (-coef)*math.log2(coef)
        information = information + first_part*logpart
    return information


data = np.array([[1, 1, 1], [1, 2, 2], [2, 2, 2], [3, 2, 2], [3, 2, 1], [2, 2, 3], [3, 1, 3], [1, 2, 2]])
labels = np.array([1, 1, 2, 2, 1, 2, 2, 1])
print(compute_information(data[:, 0], labels))



# # B E F O R E
# import random
# from anytree import Node, RenderTree
#
# from algorithms.auxiliaryFunctions import *
#
#
# class ID3Algorithm():
#
#     tst_labels = None
#
#     def __init__(self, F):
#         self.F = F
#
#     def fit(self, data, labels):
#         all_nodes_finished = False
#
#         tree = []  # Tree as an array of dictionaries (dicitionaries is a list of used nodes and indices)
#         last_added = []  # Last added nodes
#
#         # Hacer dos bucles, uno por niveles (vertical) y otro por horizontal. Usar last_added
#
#         # FIRST NODE -----------------------------------------------------------------------
#         # Compute features random for the particular node
#         vec_random = []
#         possible_feat = [it for it in range(data.shape[1])]
#         for it in range(self.F):
#             index_randnum = random.randrange(0, len(possible_feat), 1)
#             vec_random.append(possible_feat[index_randnum])
#             possible_feat.pop(index_randnum)
#
#         # Find best feature
#         information = []
#         for feature_number in vec_random:
#             information.append(compute_information(data[:,feature_number], labels))
#         best_indx = information.index(min(information))
#         best_attr = vec_random[best_indx]
#         keyy = '['+str(best_attr)+']'
#         dictionary = [keyy,[it for it in range(data.shape[0])],[]]
#         tree.append(dictionary)
#         last_added.append(dictionary)
#
#         # REST OF THE NODES ----------------------------------------------------------------
#         while len(last_added) > 0:
#             new_added = []
#
#             for last_add in last_added:
#
#                 # Divide leaf with its children
#                 att_values, att = number_differentvalues_att(last_add, data)
#                 Natt_value = len(att_values)
#                 dict_inst_satis = instances_satisfying_attvalue(data[last_add[1],att], last_add[1])
#
#                 # For each children
#                 for i in range(Natt_value):
#                     # Compute features random for the particular node (that have not been used and that are differ.)
#                     vec_random = []
#                     used_feat = compute_used_features(last_add[0])
#                     possible_feat = [it for it in range(data.shape[1]) if (it not in used_feat)]
#                     for it in range(self.F):
#                         index_randnum = random.randrange(0, len(possible_feat), 1)
#                         vec_random.append(possible_feat[index_randnum])
#                         possible_feat.pop(index_randnum)
#
#                     # Find best feature
#                     information = []
#                     indices = dict_inst_satis[att_values[i]]
#                     for feature_number in vec_random:
#                         information = information.append(compute_information(data[indices,feature_number],
#                                                                              labels[last_add[1]],))
#                     best_indx = information.index(min(information))
#                     best_attr = vec_random[best_indx]
#                     keyy = '[' + str(best_attr) + ']'
#                     dictionary = [last_add[0]+keyy, indices, tree[last_add[0]].append(att_values[i])]
#
#                     # Find if the new node is terminal or not
#                     bol_terminal, lab_class = is_terminal(dictionary[0], data[indices,:], labels[indices])
#                     if bol_terminal == False:
#                         new_added.append(dictionary)
#                     else:
#                         dictionary[1] = 'Class:'+str(lab_class)
#
#                     # Add node to the tree
#                     tree.append(dictionary)
#
#             last_added = new_added
#
#
#     def classify(self, data_test):
#         pass