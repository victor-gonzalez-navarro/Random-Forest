import random

from algorithms.auxiliaryFunctions import *
from algorithms.ID3Algorithm import *
from collections import Counter


class randomForestAlgorithm():

    tst_labels = None
    trees = []
    intances_used = []

    def __init__(self, NT, F):
        self.NT = NT
        self.F = F

    def fit(self, data, labels):
        Ninstances = data.shape[0]
        # random.seed(30)

        # For b = 1, ..., NT
        for b in range(self.NT):
            # 1. Sample, with replacement, n training examples from X, Y
            vec_random = [random.randrange(0, Ninstances, 1) for it in range(Ninstances)]
            data_b = data[vec_random, :]
            labels_b = labels[vec_random]
            self.intances_used = self.intances_used + [vec_random,]

            # 2. Train a classification or regression tree on those examples
            ID3tree = ID3Algorithm(self.F, b)
            ID3tree.fit(data_b, labels_b)
            self.trees = self.trees + [ID3tree,]

    def classify(self, data_test):
        result = np.zeros((self.NT,data_test.shape[0]))
        self.tst_labels = np.zeros(data_test.shape[0])
        for b in range(self.NT):
            self.trees[b].classify(data_test, self.intances_used[b])
            result[b,:] = self.trees[b].tst_labels
        for ex in range(data_test.shape[0]):
            c = Counter(result[:,ex])
            index = 0
            lista = c.most_common(len(c))
            while (index<len(lista)-1) and (lista[index][0] == -1):
                index = index + 1
            res = lista[index][0]
            self.tst_labels[ex] = res