import random

from algorithms.auxiliaryFunctions import *
from algorithms.ID3Algorithm import *


class randomForestAlgorithm():

    tst_labels = None

    def __init__(self, NT, F):
        self.NT = NT
        self.NT = F

    def fit(self, data, labels):
        Ninstances = data.shape[0]

        # For b = 1, ..., B
        for b in range(self.NT):
            # 1. Sample, with replacement, n training examples from X, Y
            vec_random = [random.randrange(0, Ninstances, 1) for it in range(Ninstances)]
            data_b = data[vec_random, :]
            labels_b = labels[vec_random]

            # 2. Train a classification or regression tree on those examples
            ID3tree = ID3Algorithm(self.F)
            ID3tree.fit(data_b, labels_b)

    def classify(self, data_test):
        pass