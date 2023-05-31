#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# set the random state for reproducibility
import numpy as np

np.random.seed(401)
CLASSIFIER_NAMES = ['sgd_classifer', 'gaussianNB_classifier', 'random_forest_classifer',
                   'mlp_classifer', 'ada_boost_classifer']

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C) / np.sum(C)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recalls = []
    for k in range(C.shape[0]):
        recalls.append(C[k, k] / np.sum(C[k, :]))
    return recalls


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precisions = []
    for k in range(C.shape[0]):
        precisions.append(C[k, k] / np.sum(C[:, k]))
    return precisions


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
    Returns:
       i: int, the index of the supposed best classifier
    '''
    print("doing class 31 task: ...")
    classifiers = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05), AdaBoostClassifier()]

    with open(f"{output_dir}/a1_3.1.txt", "w") as f:
        # For each classifier, compute results and write the following output:
        iBest = 1
        best_acc = 0
        for i, clf in enumerate(classifiers):
            name = CLASSIFIER_NAMES[i]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)

            if acc > best_acc:
                best_acc = acc
                iBest = i
            f.write(f'Results for {name}:\n')  # Classifier name
            print(f"Accuracy: {acc}")
            f.write(f'\tAccuracy: {acc:.4f}\n')
            print(f"Recall: {rec}")
            f.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            print(f"Precision: {prec}")
            f.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            f.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    print(f"the best accuracy is {CLASSIFIER_NAMES[iBest]}")
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)
    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print("doing class 32 task: ...")
    classifiers = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05), AdaBoostClassifier()]
    X_1k = None
    y_1k = None
    with open(f"{output_dir}/a1_3.2.txt", "w") as f:
        sample_sizes = [1000, 5000, 10000, 15000, 20000]
        clf = classifiers[iBest]
        name = CLASSIFIER_NAMES[iBest]

        for s in sample_sizes:
            rand_data = np.random.randint(0, X_train.shape[0], s)
            X_train_new = X_train[rand_data, :]
            y_train_new = y_train[rand_data]
            print(f"Training use {name} and Data size: {s}")
            clf.fit(X_train_new, y_train_new)
            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            acc = accuracy(conf_matrix)
            print("Accuracy")
            print(acc)
            f.write(f'{s}: {round(acc, 4)}\n')

            if s == 1000:
                X_1k = X_train_new
                y_1k = y_train_new

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print("doing class 33 task: ...")
    classifiers = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05), AdaBoostClassifier()]
    clf = classifiers[i]
    name = CLASSIFIER_NAMES[i]

    # 5 features, 32k data
    selector_32k_5 = SelectKBest(f_classif, k=5)
    X_32k_5 = selector_32k_5.fit_transform(X_train, y_train)
    p_32k_5 = selector_32k_5.pvalues_
    feat_32k_5 = selector_32k_5.get_support(indices=True)
    clf.fit(X_32k_5, y_train)
    y_pred_32k_5 = clf.predict(selector_32k_5.transform(X_test))
    conf_32k_5 = confusion_matrix(y_test, y_pred_32k_5)
    acc_32k_5 = accuracy(conf_32k_5)

    # 50 features, 32k data
    selector_32k_50 = SelectKBest(f_classif, k=50)
    X_32k_50 = selector_32k_50.fit_transform(X_train, y_train)
    p_32k_50 = selector_32k_50.pvalues_
    feat_32k_50 = selector_32k_50.get_support(indices=True)

    # 5 features, 1k data
    selector_1k_5 = SelectKBest(f_classif, k=5)
    X_1k_5 = selector_1k_5.fit_transform(X_1k, y_1k)
    p_1k_5 = selector_1k_5.pvalues_
    feat_1k_5 = selector_1k_5.get_support(indices=True)
    clf.fit(X_1k_5, y_1k)
    y_pred_1k_5 = clf.predict(selector_1k_5.transform(X_test))
    conf_1k_5 = confusion_matrix(y_test, y_pred_1k_5)
    acc_1k_5 = accuracy(conf_1k_5)

    feature_intersection = list(set(feat_32k_5) & set(feat_1k_5))
    top5 = feat_32k_5

    with open(f"{output_dir}/a1_3.3.txt", "w") as f:
        f.write(f'{5} p-values: {[round(pval, 4) for pval in p_32k_5]}\n')
        f.write(f'{50} p-values: {[round(pval, 4) for pval in p_32k_50]}\n')
        f.write(f'Accuracy for 1k: {acc_1k_5:.4f}\n')
        f.write(f'Accuracy for full dataset: {acc_32k_5:.4f}\n')
        f.write(f'Chosen feature intersection: {feature_intersection}\n')
        f.write(f'Top-5 at higher: {top5}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''

    print("doing class 34 task: ...")
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((y_train, y_test))
    kf = KFold(n_splits=5, shuffle=True)

    classifiers = [SGDClassifier(), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05), AdaBoostClassifier()]

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        accuracies = []
        for train_index, test_index in kf.split(X, Y):
            kf_accuracies = []

            for j, clf in enumerate(classifiers):
                clf.fit(X[train_index], Y[train_index])
                y_pred = clf.predict(X[test_index])
                conf_matrix = confusion_matrix(Y[test_index], y_pred)
                kf_accuracies.append(accuracy(conf_matrix))
            print("[KFold] Accuracies: " + str(kf_accuracies))
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kf_accuracies]}\n')
            accuracies.append(kf_accuracies)

        p_values = []
        for index, clf in enumerate(classifiers):
            if index != i:
                s = ttest_rel(accuracies[index], accuracies[i])
                p_values.append(s.pvalue)
                print(f"{CLASSIFIER_NAMES[index]} compares with {CLASSIFIER_NAMES[i]} has p value: {s.pvalue}")
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    data = np.load(args.input)['arr_0']
    X = data[:, :173]
    Y = data[:, 173]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)



    # TODO : complete each classification experiment, in sequence.
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)