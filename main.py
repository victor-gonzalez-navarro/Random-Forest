import numpy as np
import random
import math

from algorithms.preprocessing import preprocess
from algorithms.randomForestAlgorithm import randomForestAlgorithm


# ----------------------------------------------------------------------------------------------------------------- Main
def main():

    for itt0 in range(1,6):
        dta_option1 = itt0
        rows, labels = preprocess(dta_option1)
        data = np.array(rows)
        data = data.astype(np.float)
        labels = (np.array(labels)).reshape(data.shape[0])
        labels = labels.astype(np.int)
        M = data.shape[1]
        for itt1 in [50,100]:
            NT = itt1
            for itt2 in [1, 3, int(math.log(M,2)+1), int(math.sqrt(M))]:
                F = itt2
                if data.shape[1] < F:
                    F = data.shape[1]
                indicc = np.arange(data.shape[0])
                np.random.shuffle(indicc)
                data = np.array(data)
                data_trn = data[indicc[:], :]
                labels_trn = labels[indicc[:]]

                randForest = randomForestAlgorithm(NT, F)
                randForest.fit(data_trn, labels_trn)
                randForest.classify(data_trn)

                len_test_labels = sum((randForest.tst_labels > (-1)) == True)
                test_accuracy = (sum([a == b for a,b in zip(labels_trn, randForest.tst_labels)]))/len_test_labels
                print('\033[1m' + 'The number of the dataset selected is: ' + str(dta_option1) + '\033[0m')
                print('\033[1m' + 'The number of Trees is: ' + str(NT) + '\033[0m')
                print('\033[1m' + 'The number of Features in the Forest is: ' + str(F) + '\033[0m')
                print('\033[1m' + '**The final test accuracy is: ' + str(round(test_accuracy, 3)) +'**'+ '\033[0m')
                print('--------------------------------------------------------------')


    # # HYPERPARAMETERS **********************************
    # dta_option1 = 3  # Number of the Dataset
    # NT = 10  # Number of trees
    # F = 3  # Number of features
    # # random.seed(30)
    # # **************************************************

    # print('\033[1m' + 'The number of the dataset selected is: ' + str(dta_option1) + '\033[0m')
    # print('\033[1m' + 'The number of Trees is: ' + str(NT) + '\033[0m')
    # print('\033[1m' + 'The number of Features in the Forest is: ' + str(F) + '\033[0m')
#
#
    # # Preprocess the data
    # rows, labels = preprocess(dta_option1)
    # data = np.array(rows)
    # data = data.astype(np.float)
    # labels = (np.array(labels)).reshape(data.shape[0])
    # labels = labels.astype(np.int)
    # if data.shape[1] < F:
    #     F = data.shape[1]
#
    # # Shuffle data, and divide train and test set
    # indicc = np.arange(data.shape[0])
    # np.random.shuffle(indicc)
    # # random.Random(6).shuffle(indicc)  # so as to replicate results (seed)
    # data = np.array(data)
    # data_trn = data[indicc[:], :]
    # labels_trn = labels[indicc[:]]
#
    # # Call fit and predict of the algorithm (RF) I have implemented
    # randForest = randomForestAlgorithm(NT, F)
    # randForest.fit(data_trn, labels_trn)
    # randForest.classify(data_trn)  # we compute out of bag error
#
    # # Compute Accuracy between ground truth and predicted labels
    # len_test_labels = sum((randForest.tst_labels>(-1)) == True)
    # test_accuracy = (sum([a == b for a,b in zip(labels_trn, randForest.tst_labels)]))/len_test_labels
    # print('\033[1m' + '**The final test accuracy is: ' + str(round(test_accuracy, 3)) +'**'+ '\033[0m')

# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()