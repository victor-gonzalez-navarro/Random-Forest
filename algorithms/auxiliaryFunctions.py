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
    N_classes = len(set(labels))
    N_instances = len(labels)

    classes_vector = [0.0] * N_classes
    for it in range(N_instances):
        item = data[it]
        if item not in dat:
            new_class_vector = np.copy(classes_vector)
            new_class_vector[labels[it]-1] = new_class_vector[labels[it]-1] + 1.0
            dat[item] = [1.0, new_class_vector]
        else:
            dat[item][0] = dat[item][0] + 1.0
            dat[item][1][labels[it]-1] = dat[item][1][labels[it]-1] + 1.0

    information = 0
    for item in dat.keys():
        logpart = 0
        dat[item][1] = np.divide(dat[item][1], sum(dat[item][1]))
        for coef in dat[item][1]:
            if coef != 0:
                logpart = logpart + (-coef) * (math.log2(coef))

        information = information + (dat[item][0] / float(N_instances)) * logpart
    return information

def number_differentvalues_att(last_add, data):
    string = last_add[0]
    it = len(string)-1
    feature = ''
    while string[it] != '[':
        feature = string[it] + feature
        it = it - 1
    result = list(set(data[last_add[1],int(feature)]))
    return result, int(feature)

def instances_satisfying_attvalue(data, att_values, inst):
    result = dict()
    for i in range(len(data)):
        keyy = data[i]
        if keyy in result:
            result[keyy] = result[keyy].append(inst[i])
        else:
            result[keyy] = [inst[i]]
    return result



