import numpy as np
import random
import math
import matplotlib.pyplot as plt

from algorithms.preprocessing import preprocess
from algorithms.randomForestAlgorithm import randomForestAlgorithm


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    # # HYPERPARAMETERS **********************************
    # dta_option1 = 1  # Number of the Dataset (4)
    # NT = 100  # Number of trees
    # F = # 3  # Number of features
    # # *************************************************
    # #random.seed(6)
    #
    # print('\033[1m' + 'The number of the dataset selected is: ' + str(dta_option1) + '\033[0m')
    # print('\033[1m' + 'The number of Trees is: ' + str(NT) + '\033[0m')
    # print('\033[1m' + 'The number of Features in the Forest is: ' + str(F) + '\033[0m')
    # print('\nComputing the forest... Please wait maximum 1 minute')

    # # Preprocess the data
    # rows, labels = preprocess(dta_option1)
    # data = np.array(rows)
    # data = data.astype(np.float)
    # labels = (np.array(labels)).reshape(data.shape[0])
    # labels = labels.astype(np.int)
    # if data.shape[1] < F:
    #     F = data.shape[1]
    # dict_Att = dict()
    # for i in range(data.shape[1]):
    #     dict_Att[i] = list(set(data[:, i]))

    # # Shuffle data, and divide train and test set
    # indicc = np.arange(data.shape[0])
    # np.random.shuffle(indicc)
    # # random.Random(6).shuffle(indicc)  # so as to replicate results (seed)
    # data = np.array(data)
    # data_trn = data[indicc[:], :]
    # labels_trn = labels[indicc[:]]

    # # Call fit and predict of the algorithm (RF) I have implemented
    # randForest = randomForestAlgorithm(NT, F)
    # randForest.fit(data_trn, labels_trn, dict_Att)
    # randForest.classify(data_trn)  # we compute the out of bag error

    # # Compute Accuracy between ground truth and predicted labels
    # len_test_labels = sum((randForest.tst_labels > (-1)) == True)  # Do not consider not classified instances
    # test_accuracy = (sum([a == b for a, b in zip(labels_trn, randForest.tst_labels)])) / len_test_labels
    # print('\033[1m' + '**The final accuracy is: ' + str(round(test_accuracy, 3)) + '**' + '\033[0m')

    dta_option1_vec = [1,2,3,4,5]
    for dta_option1 in dta_option1_vec:
        # Preprocess the data
        rows, labels = preprocess(dta_option1)
        data = np.array(rows)
        data = data.astype(np.float)
        labels = (np.array(labels)).reshape(data.shape[0])
        labels = labels.astype(np.int)

        dict_Att = dict()
        for i in range(data.shape[1]):
            dict_Att[i] = list(set(data[:, i]))

        # Shuffle data, and divide train and test set
        indicc = np.arange(data.shape[0])
        np.random.shuffle(indicc)
        # random.Random(6).shuffle(indicc)  # so as to replicate results (seed)
        data = np.array(data)
        data_trn = data[indicc[:], :]
        labels_trn = labels[indicc[:]]

        NT_vec = [50, 100]
        for NT in NT_vec:
            F_vec = [1, 3, int(math.log(data.shape[1],2)+1), int(math.sqrt(data.shape[1]))]
            counter = 0
            for F in F_vec:
                counter = counter + 1
                print('---------------------------------------------------------------------------------------')
                if data.shape[1] < F:
                    F = data.shape[1]
                print('\033[1m' + 'Number dataset: ' + str(dta_option1) + ' | Number of trees: '+str(NT)+
                                        ' | Number of Features: '+str(F)+'\033[0m')


                # Call fit and predict of the algorithm (RF) I have implemented
                randForest = randomForestAlgorithm(NT, F)
                randForest.fit(data_trn, labels_trn, dict_Att)

                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                randForest.plotFeaturesImportance(ax, counter)
                if counter == 3:
                    FF = 'log2(M)+1'
                elif counter == 4:
                    FF = 'sqrt(M)'
                else:
                    FF = str(F)
                fig.savefig('./FeatureImportance/Dataset' + str(dta_option1)+'NT='+ str(NT)+'-F='+FF+ '.png')
                randForest.classify(data_trn)  # we compute the out of bag error

                # Compute Accuracy between ground truth and predicted labels
                len_test_labels = sum((randForest.tst_labels>(-1)) == True)  # Do not consider not classified instances
                test_accuracy = (sum([a == b for a,b in zip(labels_trn, randForest.tst_labels)]))/len_test_labels
                print('\033[1m' + '**The final accuracy is: ' + str(round(test_accuracy, 3)) + '**' + '\033[0m')

        plt.rcParams['figure.constrained_layout.use'] = True
        if dta_option1 == 1:
            namedata = 'Contact Lenses'
        elif dta_option1 == 2:
            namedata = 'Iris'
        elif dta_option1 == 3:
            namedata = 'Contraceptive Method Choice'
        elif dta_option1 == 4:
            namedata = 'Liver Patient'
        elif dta_option1 == 5:
            namedata = 'Chess'

# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()