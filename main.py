import numpy as np

from algorithms.preprocessing import preprocess
from algorithms.randomForestAlgorithm import randomForestAlgorithm


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    # HYPERPARAMETERS **********************************
    dta_option1 = 0  # Number of the Dataset
    print('Change fraction train test')
    perc_test = 0  # Fraction of Test set
    NT = 5  # Number of trees
    F = 3  # Number of features
    # **************************************************

    print('\033[1m' + 'The number of the dataset selected is: ' + str(dta_option1) + '\033[0m')
    print('\033[1m' + 'The number of Trees is: ' + str(NT) + '\033[0m')
    print('\033[1m' + 'The number of Features in the Forest is: ' + str(F) + '\033[0m')

    print('********************************************** Rules induced with Train set')

    # Preprocess the data
    rows, labels = preprocess(dta_option1)
    data = np.array(rows)
    data = data.astype(np.float)
    labels = (np.array(labels)).reshape(data.shape[0])
    labels = labels.astype(np.int)
    if data.shape[1] < F:
        F = data.shape[1]

    # Shuffle data, and divide train and test set
    indicc = np.arange(data.shape[0])
    np.random.shuffle(indicc)
    cut = round(perc_test * data.shape[0])
    data = np.array(data)
    data_test = data[indicc[0:cut], :]
    labels_test = labels[indicc[0:cut]]
    data_trn = data[indicc[cut:], :]
    labels_trn = labels[indicc[cut:]]

    # Call fit and predict of the algorithm (RF) I have implemented
    randForest = randomForestAlgorithm(NT, F)
    randForest.fit(data_trn, labels_trn)
    randForest.classify(data_test)

    # Compute Accuracy between ground truth and predicted labels
    test_accuracy = (sum([a == b for a,b in zip(labels_test, randForest.tst_labels)]))/len(labels_test)
    print('\033[1m' + 'The final test accuracy is: ' + str(round(test_accuracy, 3)) + '\033[0m')


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()