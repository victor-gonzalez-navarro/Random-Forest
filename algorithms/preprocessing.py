import numpy as np
import csv

from sklearn import preprocessing
from collections import Counter


def preprocess(dta_option1):

    if dta_option1 == 1:
        # Contact Lenses dataset
        file = open('./datasets/lenses.data.csv', 'r')
    elif dta_option1 == 2:
        # Iris dataset
        file = open('./datasets/iris.csv', 'r')
    elif dta_option1 == 3:
        # Primary Tumor dataset
        file = open('./datasets/primary-tumor.data.csv', 'r')
    elif dta_option1 == 4:
        # Dataset obtained in https://www.openml.org/d/1480
        file = open('./datasets/liverPatientDataset.csv', 'r')
    elif dta_option1 == 5:
        # Dataset obtained in https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
        file = open('./datasets/kr-vs-kp.data.csv', 'r')

    rows = []
    labels = []
    if (dta_option1==1):
        for line in file:
            row = line.split()
            rows.append(row[1:-1])
            labels.append(row[-1:])

    elif (dta_option1 == 2):
        for line in file:
            row = line.split(',')
            rows.append(row[0:-1])
            labels.append(row[-1:][0][0])
        rows = np.array(rows)
        rows = rows.astype(np.float)
        rows_min = np.min(rows, axis=0)
        rows = rows - rows_min
        rows_max = np.max(rows, axis=0)
        rows = rows / rows_max
        a = input('Falta acabar preprocess')

    elif (dta_option1 == 3):
        for line in file:
            line = line[:-1]
            row = line.split(',')
            cnt = Counter(row[1:])
            moda = cnt.most_common(1)[0][0]
            for item in range(len(row)):
                if row[item] == '?':
                    row[item] = moda
            rows.append(row[1:])
            delee = row[0]
            labels.append(row[0])
        # Transform to label encoding
        flatten = set([item for sublist in rows for item in sublist])
        le = preprocessing.LabelEncoder()
        le.fit(list(flatten))
        for line in range(len(rows)):
            rows[line] = le.transform(rows[line])
        a = input('Falta acabar preprocess')



    elif (dta_option1 == 4):
        with file as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                rows.append(row[0:-1])
                labels.append(row[-1:][0][0])

        flatten = set([item for sublist in rows for item in sublist])
        le = preprocessing.LabelEncoder()
        le.fit(list(flatten))
        for line in range(len(rows)):
            for att in range(len(rows[line])):
                if numoricalatt[att] == 0:
                    rows[line][att] = le.transform([rows[line][att]])[0]
        rows = np.array(rows)
        rows = rows.astype(np.float)
        for att in range(len(numoricalatt)):
            if numoricalatt[att] == 1:
                rows[:,att] = rows[:,att]-np.min(rows[:,att])
                rows[:,att] = rows[:,att]/np.max(rows[:,att])
        flatten = set(labels)
        le.fit(list(flatten))
        labels = le.transform(labels)
        a = input('Falta acabar preprocess')

    elif (dta_option1==5):
        for line in file:
            row = line.split(',')
            rows.append(row[0:-1])
            labels.append(row[-1:][0][0])
        # Transform to label encoding
        flatten = set([item for sublist in rows for item in sublist])
        le = preprocessing.LabelEncoder()
        le.fit(list(flatten))
        for line in range(len(rows)):
            rows[line] = le.transform(rows[line])
        flatten = set(labels)
        le.fit(list(flatten))
        labels = le.transform(labels)
        a = input('Falta acabar preprocess')

    return rows, labels

