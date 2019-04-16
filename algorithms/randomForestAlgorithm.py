import random

from algorithms.auxiliaryFunctions import *
from algorithms.ID3Algorithm import *
from collections import Counter


class randomForestAlgorithm():

    tst_labels = None
    trees = []

    def __init__(self, NT, F):
        self.NT = NT
        self.F = F

    def fit(self, data, labels):
        Ninstances = data.shape[0]

        # For b = 1, ..., NT
        for b in range(self.NT):
            # 1. Sample, with replacement, n training examples from X, Y
            vec_random = [random.randrange(0, Ninstances, 1) for it in range(Ninstances)]
            data_b = data[vec_random, :]
            labels_b = labels[vec_random]

            # 2. Train a classification or regression tree on those examples
            ID3tree = ID3Algorithm(self.F)
            ID3tree.fit(data_b, labels_b)
            self.trees = self.trees + [ID3tree,]

    def classify(self, data_test):
        result = np.zeros((self.NT,data_test.shape[0]))
        self.tst_labels = np.zeros(data_test.shape[0])
        for b in range(self.NT):
            self.trees[b].classify(data_test)
            result[b,:] = self.trees[b].tst_labels
        for ex in range(data_test.shape[0]):
            c = Counter(result[:,ex])
            res = c.most_common(1)[0][0]
            self.tst_labels[ex] = res