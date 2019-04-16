import numpy as np
import math


# -------------------------------------------------------------------------------------------------- AUXILIARY FUNCTIONS
def compute_used_features(string):
    features = []
    feat = ''
    for letter in string:
        if letter != '[' and letter != ']':
            feat = feat + letter
        if letter == ']':
            features.append(int(feat))
            feat = ''
    return features


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


def number_differentvalues_att(last_add, data):
    string = last_add[0]
    it = len(string)-2
    feature = ''
    while string[it] != '[':
        feature = string[it] + feature
        it = it - 1
    result = list(set(data[last_add[1],int(feature)]))
    return result, int(feature)


def instances_satisfying_attvalue(data, inst):
    result = dict()
    for i in range(len(data)):
        keyy = data[i]
        if keyy in result:
            result[keyy] = result[keyy] + [inst[i],]
        else:
            result[keyy] = [inst[i]]
    return result


def is_terminal(string, data, labels):
    isterminal = False
    label = None
    # All instances belong to the same class?
    if len(set(labels))==1:
        isterminal = True
        label = labels[0]
    # There are no more attribute lefts to use
    elif len(compute_used_features(string)) == data.shape[1]:
        isterminal = True
        # The label assigned is the majority of the instances in the leaf
        label = labels
    return isterminal, label

def draw_tree(tree):
    file = open('./testfile.txt', 'w')
    file.write('digraph G {')
    enter = False
    finito = 0
    for item in tree.keys():
        if enter:
            if 'Class' not in item:
                phrase = tree[item][2]
                it = len(phrase) - 1
                label = ''
                while phrase[it] != '*':
                    label = phrase[it] + label
                    it = it - 1
                path = compute_used_features(item)
                file.write('\n "At: '+str(path[-2])+ '" -> ' +'"At: '+str(path[-1]) + '" [label='+label+']')
            else:
                finito = finito + 1
                it = len(item) - 1
                label = ''
                while item[it] != '*':
                    label = item[it] + label
                    it = it - 1
                while item[it] != '-':
                    it = it - 1
                it = it - 2
                class_label = ''
                while item[it] != ':':
                    class_label = item[it] + class_label
                    it = it - 1
                while item[it] != ']':
                    it = it - 1
                letter = ''
                while item[it] != '[':
                    letter = item[it] + letter
                    it = it - 1
                letter = '[' + letter
                path = compute_used_features(letter)
                file.write('\n "At: '+str(path[0])+ '" -> ' +'"T'+str(finito)+': Class '+ str(
                    class_label) + '" [label='+label+']')
        enter = True
    file.write('\n}')
    file.close()




