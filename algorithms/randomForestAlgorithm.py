import random
import matplotlib.pyplot as plt

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

    def fit(self, data, labels, dict_Att):
        Ninstances = data.shape[0]
        # random.seed(35)

        # For b = 1, ..., NT
        for b in range(self.NT):
            # 1. Sample, with replacement, n training examples from X, Y
            vec_random = [random.randrange(0, Ninstances, 1) for it in range(Ninstances)]
            data_b = data[vec_random, :]
            labels_b = labels[vec_random]
            self.intances_used = self.intances_used + [vec_random,]

            # 2. Train a classification or regression tree on those examples
            ID3tree = ID3Algorithm(self.F, b)
            ID3tree.fit(data_b, labels_b, dict_Att)
            self.trees = self.trees + [ID3tree,]

    def plotFeaturesImportance(self, ax, counter):
        appearance_features = []
        for b in range(self.NT):
            appearance_features = appearance_features + self.trees[b].best_features_used
        d = Counter(appearance_features)
        deg, cnt = zip(*d.items())
        print('Starting from zero, the importance of each feture ordered from wort to best (ascending order) is:')
        order = np.argsort(cnt)
        print(np.array(deg)[order])
        cnt = np.true_divide(np.array(cnt), len(appearance_features))
        ax.bar(deg, cnt, width=len(deg)/(2*len(deg)), color='b')
        if counter == 3:
            ax.title.set_text("Importance [NT="+str(self.NT)+', F=(log2(M)+1)]')
        elif counter == 4:
            ax.title.set_text("Importance [NT="+str(self.NT)+', F=sqrt(M)]')
        else:
            ax.title.set_text("Importance [NT="+str(self.NT)+', F='+str(self.F)+']')
        ax.set_xlabel("Feature number")
        ax.set_ylabel("Frequency")

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