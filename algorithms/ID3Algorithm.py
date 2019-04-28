import random

from algorithms.auxiliaryFunctions import *
from collections import Counter


class ID3Algorithm():

    tst_labels = None
    tree = None

    def __init__(self, F, b):
        self.F = F
        self.b = b

    def fit(self, data, labels):
        # random.seed(15)

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
        tree[dictionary[0]+dictionary[2]] = dictionary  # dictionary is a list with: the attributes seen, the instances that are included, and the value o the attributes (i.e., useful for the tree)
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
                        dictionary = [last_add[0]+keyy, indices, tree[last_add[0]+last_add[2]][2]+'*'+str(att_values[i])]

                        # Add node to analyze
                        new_added.append(dictionary)
                        # Add node to the tree
                        tree[dictionary[0] + dictionary[2]] = dictionary
                    else:
                        sequence = tree[last_add[0]+last_add[2]][2]+'*'+str(att_values[i])
                        if isinstance(lab_class,np.ndarray):
                            c = Counter(lab_class)
                            best = c.most_common(1)[0][0]
                            str_lab_class = str(best)+' Acc. '+str(round(sum(lab_class==best)/len(lab_class),2))
                        else:
                            str_lab_class = str(lab_class)


                        dictionary = [last_add[0]+'--Class:'+str_lab_class+'--Seq:'+sequence, dict_inst_satis[
                            att_values[i]], sequence]
                        tree[dictionary[0]] = dictionary


            last_added = new_added
        # draw_tree(tree, self.b)
        self.tree = tree

    def classify(self, data_test, intances_used):
        test_labels = []
        for i in range(data_test.shape[0]):
            if (i in intances_used):
                test_labels.append(-1)
            else:
                for itemm in self.tree.keys():
                    item = self.tree[itemm][0]
                    if 'Class' in item:
                        pointer2 = 0
                        while item[pointer2] != ':':
                            pointer2 = pointer2 + 1
                        pointer2 = pointer2 + 1
                        class_l = ''
                        while item[pointer2] != '-':
                            class_l = class_l + str(item[pointer2])
                            pointer2 = pointer2 + 1
                        while item[pointer2] != '*':
                            pointer2 = pointer2 + 1
                        pointer2 = pointer2 + 1
                        equal = True
                        while equal:
                            # Determine feature
                            feat_num = ''
                            pointer1 = 1
                            while item[pointer1] != ']':
                                feat_num = feat_num + str(item[pointer1])
                                pointer1 = pointer1 + 1
                            feat_num = int(float(feat_num))

                            # Determine value
                            val_num = ''
                            while (pointer2 < (len(item))) and (item[pointer2] != '*'):
                                val_num = val_num + str(item[pointer2])
                                pointer2 = pointer2 + 1
                            pointer2 = pointer2 + 1
                            if val_num == '':
                                equal = False
                                test_labels.append(float(class_l))
                            # Equality?
                            elif data_test[i,feat_num] != float(val_num):
                                equal = False
                if len(test_labels) == i:
                    test_labels.append(-1)

        self.tst_labels = test_labels