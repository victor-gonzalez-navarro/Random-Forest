import numpy as np
import csv

from sklearn import preprocessing
from collections import Counter


def preprocess(dta_option1):

    if dta_option1 == 0:
        # Slides dataset
        file = open('./datasets/slides.csv', 'r')
    if dta_option1 == 1:
        # Contact Lenses dataset
        file = open('./datasets/lenses.data.csv', 'r')
    elif dta_option1 == 2:
        # Iris dataset
        file = open('./datasets/iris.csv', 'r')
    elif dta_option1 == 3:
        # Contraceptive Method Choice Data Set
        file = open('./datasets/cmc.csv', 'r')
    elif dta_option1 == 4:
        # Dataset obtained in https://www.openml.org/d/1480
        file = open('./datasets/liverPatientDataset.csv', 'r')
    elif dta_option1 == 5:
        # Dataset obtained in https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
        file = open('./datasets/kr-vs-kp.data.csv', 'r')

    rows = []
    labels = []

    # Slides dataset (8 instances)
    if (dta_option1==0):
        for line in file:
            row = line.split()
            rows.append(row[1:-1])
            labels.append(row[-1:])

    # Contact lenses dataset (24 instances)
    elif (dta_option1==1):
        for line in file:
            row = line.split()
            rows.append(row[1:-1])
            labels.append(row[-1:])

    # Iris dataset (150 instances)
    elif (dta_option1 == 2):
        n_bins = 3
        for line in file:
            row = line.split(',')
            rows.append(row[0:-1])
            labels.append(row[-1:][0][0])
        rows = np.array(rows)
        rows = rows.astype(np.float)
        for id in range(rows.shape[1]):
            bins = np.linspace(min(rows[:,id])-0.05*min(rows[:,id]), max(rows[:,id]), num=n_bins+1)
            rows[:, id] = np.digitize(rows[:,id], bins, right=True)

    # Contraceptive Method Choice Data Set (1473 instances)
    elif (dta_option1==3):
        n_bins = 3
        for line in file:
            row = line.split(',')
            rows.append(row[0:-1])
            labels.append(row[-1:][0][0])
        # Transform to label encoding
        rows = np.array(rows)
        for id in range(rows.shape[1]):
            if id not in [0,3]:
                flatten = set(rows[:,id])
                le = preprocessing.LabelEncoder()
                le.fit(list(flatten))
                rows[:, id] = le.transform(rows[:,id])
        rows = rows.astype(np.float)
        for id in [0, 3]:
            rows[:,id] = rows[:,id].astype(np.float)
            bins = np.linspace(min(rows[:, id]) - 0.1, max(rows[:, id]), num=n_bins + 1)
            rows[:, id] = np.digitize(rows[:, id], bins, right=True)
        for id1 in range(rows.shape[0]):
            for id2 in range(rows.shape[1]):
                if id2 not in [0, 3]:
                    rows[id1, id2] = rows[id1, id2] + 1
        flatten = set(labels)
        le.fit(list(flatten))
        labels = le.transform(labels)
        for id in range(len(labels)):
            labels[id] = labels[id] + 1

    # Liver Patient dataset (583 instances)
    elif (dta_option1 == 4):
        n_bins = 9
        with file as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                rows.append(row[0:-1])
                labels.append(row[-1:][0][0])

        for line in range(len(rows)):
            for att in range(len(rows[line])):
                if att == 1:
                    flatten = set(['Male','Female'])
                    le = preprocessing.LabelEncoder()
                    le.fit(list(flatten))
                    rows[line][att] = le.transform([rows[line][att]])[0]
        rows = np.array(rows)
        rows = rows.astype(np.float)
        for id in range(rows.shape[0]):
            rows[id,1] = rows[id,1] + 1
        for id in range(rows.shape[1]):
            if id != 1:
                bins = np.linspace(min(rows[:,id])-0.05*min(rows[:,id]), max(rows[:,id]), num=n_bins+1)
                rows[:, id] = np.digitize(rows[:,id], bins, right=True)
        flatten = set(labels)
        le.fit(list(flatten))
        labels = le.transform(labels)
        for id in range(len(labels)):
            labels[id] = labels[id] + 1

    # Chess dataset (3196 instances)
    elif (dta_option1==5):
        for line in file:
            row = line.split(',')
            rows.append(row[0:-1])
            labels.append(row[-1:][0][0])
        # Transform to label encoding
        rows = np.array(rows)
        for id in range(rows.shape[1]):
            flatten = set(rows[:,id])
            le = preprocessing.LabelEncoder()
            le.fit(list(flatten))
            rows[:, id] = le.transform(rows[:,id])
        rows = rows.astype(np.float)
        for id1 in range(rows.shape[0]):
            for id2 in range(rows.shape[1]):
                rows[id1, id2] = rows[id1, id2] + 1
        flatten = set(labels)
        le.fit(list(flatten))
        labels = le.transform(labels)
        for id in range(len(labels)):
            labels[id] = labels[id] + 1

    return rows, labels

