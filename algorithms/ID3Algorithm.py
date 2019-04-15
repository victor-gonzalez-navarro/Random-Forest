import random
from anytree import Node, RenderTree

from algorithms.auxiliaryFunctions import *


class ID3Algorithm():

    tst_labels = None

    def __init__(self, F):
        self.F = F

    def fit(self, data, labels):
        all_nodes_finished = False

        tree = []  # Tree as an array of dictionaries (dicitionaries is a list of used nodes and indices)
        last_added = []  # Last added nodes

        # Hacer dos bucles, uno por niveles (vertical) y otro por horizontal. Usar last_added

        # FIRST NODE ----------------------------------------------------------------
        vec_random = [random.randrange(0, self.F, 1) for it in range(self.F)]
        for feature_number in vec_random:
            information = information + [compute_information(data, [], labels, feature_number), ]
        best_indx = information.index(min(information))
        best_attr = vec_random[best_indx]
        keyy = '['+str(best_attr)+']'
        dictionary = [keyy,[it for it in range(data.shape[0])]]
        tree.append(dictionary)
        last_added.append(dictionary)

        while len(last_added) > 0:
            new_added = []

            for last_add in last_added:
                Natt_value = number_differentvalues_att(last_add[something])

                if is_terminal(last_add) == False:
                    for i in range(Natt_value):

                        # Compute features random for the particular node
                        vec_random = []
                        used_feat = compute_used_features(last_add[0])
                        for it in range(self.F):
                            randnum = random.randrange(0, self.F, 1)
                            if randnum not in used_feat:
                                vec_random = vec_random +[randnum,]

                        # Find best feature
                        for feature_number in vec_random:
                            information = information + [compute_information(data, last_add[1], labels, feature_number), ]
                        best_indx = information.index(min(information))
                        best_attr = vec_random[best_indx]
                        keyy = '[' + str(best_attr) + ']'
                        inst_satis = instances_satisfying_attvalue(best_attr, last_add[1])
                        dictionary = [last_add[0]+keyy, inst_satis]
                        tree.append(dictionary)
                        new_added.append(dictionary)

            last_added = new_added









    def classify(self, data_test):
        pass