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

def compute_used_values(string):
    values = []
    val = ''
    enter = 0
    for letter in string:
        if enter > 0:
            if letter != '*':
                val = val + letter
            if ((letter == '*') or (enter == (len(string)-1))) and ('Class' not in val):
                values.append(float(val))
                val = ''
        enter = enter + 1
    return values

def compute_used_values2(string):
    pointer = 0
    while string[pointer] != '*':
        pointer = pointer + 1

    string2 = string[pointer:]
    values = []
    val = ''
    enter = 0
    for letter in string2:
        if enter > 0:
            if letter != '*':
                val = val + letter
            if ((letter == '*') or (enter == (len(string2)-1))) and ('Class' not in val):
                values.append(float(val))
                val = ''
        enter = enter + 1
    return values

def generate_string(vector):
    string = ''
    for itt in vector:
        string = string + '[' + str(itt) + ']'
    return string

def generate_string_values(vector):
    string = ''
    for itt in vector:
        string = string + '*' + str(itt)
    return string


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
                logpart = logpart + (-coef)*math.log(coef,2)
        information = information + first_part*logpart
    return information


def number_differentvalues_attr(last_add, dict_Att):
    string = last_add[0]
    it = len(string)-2
    feature = ''
    while string[it] != '[':
        feature = string[it] + feature
        it = it - 1
    result = dict_Att[int(feature)]
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

def draw_tree(tree, b):
    file = open('./trees/testfile'+str(b)+'.txt', 'w')
    file.write('digraph G {')
    enter = False
    finito = 0
    for itemm in tree.keys():
        item = tree[itemm][0]
        if enter:
            if 'Class' not in item:
                phrase = tree[itemm][2]
                it = len(phrase) - 1
                label = ''
                while phrase[it] != '*':
                    label = phrase[it] + label
                    it = it - 1
                path = compute_used_features(itemm)
                string_new = generate_string(path[:-1])
                path2 = compute_used_values(tree[itemm][2])
                string_new2 = generate_string_values(path2[:-1])
                file.write('\n "At: '+str(path[-2])+ "\\n"+'#inst. = '+str(len(tree[string_new+string_new2][1]))+'" -> ' +'"At: '+str(path[-1]) + "\\n"+'#inst. = '+str(len(tree[itemm][1]))+'" [label='+label+']')
            else:
                finito = finito + 1
                it = len(item) - 1
                # Compute Attribute value
                label = ''
                while item[it] != '*':
                    label = item[it] + label
                    it = it - 1

                # Compute class label
                while item[it] != '-':
                    it = it - 1
                it = it - 2
                class_label = ''
                while item[it] != ':':
                    class_label = item[it] + class_label
                    it = it - 1

                # Compute last feature
                while item[it] != ']':
                    it = it - 1
                letter = ''
                while item[it] != '[':
                    letter = item[it] + letter
                    it = it - 1
                letter = '[' + letter
                path = compute_used_features(letter)

                # Compute all chain of fetures
                new_pint = 0
                ppath = ''
                while item[new_pint] != '-':
                    ppath = ppath + item[new_pint]
                    new_pint = new_pint + 1
                ppath = compute_used_features(ppath)
                string_new = generate_string(ppath)
                ppath2 = compute_used_values2(item)
                string_new2 = generate_string_values(ppath2[:-1])

                # Print
                file.write('\n "At: '+str(path[0])+ "\\n"+'#inst. = '+str(len(tree[string_new+string_new2][1]))+'" -> ' +'"T'+str(
                    finito)+': Class '+ str(class_label) + "\\n"+'#inst. = '+str(len(tree[item][1]))+ '" [label='+label+']')

        enter = True
    file.write('\n}')
    file.close()




