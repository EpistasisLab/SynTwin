import random
import scipy
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics

#path_to_mc-medgan: synthetic_algorithms_comparison/step2_synthetic_algorithms/MC-MedGAN

np.random.seed(123)

data_a = pd.read_csv('/path_to_mc-medgan/data/seerbreast/encoded_data.txt')
data_a = data_a.iloc[: , 2:]

data_b = pd.read_csv('/path_to_mc-medgan/samples/mc-medgan/seerbreast/sample.csv')
data_b = data_b.drop(columns = ["PatientID"])

print(data_a.shape)
print(data_b.shape)

def align_columns(df_r, df_s):
    """ Helper function to make sure two 
        dataFrames have their columns aligned. """
    df_r.sort_index(axis=1, inplace=True)
    df_s.sort_index(axis=1, inplace=True)

    def checkEqual(L1, L2):
        return len(L1) == len(L2) and sorted(L1) == sorted(L2)
    assert checkEqual(df_r.columns.tolist(), df_s.columns.tolist())
    return df_r, df_s

def cross_classification(data_a, data_b, 
                         base_metric='accuracy',
                         class_method='DT'):
    """ Classification error normalized by the error
        in the hold-out (real) set. """

    data_a, data_b = align_columns(data_a, data_b)

    # data_a: real dataset
    # data_b: synthetic dataset
    shfl_ids = np.random.permutation(range(data_a.shape[0]))
    n_train = int(data_a.shape[0] * 0.6)

    accs = dict()  # stores performances for all columns

    for col_k in data_a.columns:
        x_train = data_a.loc[shfl_ids[:n_train],
                             data_a.columns != col_k].values
        y_train = data_a.loc[shfl_ids[:n_train],
                             [col_k]].values.ravel()

        x_test = data_a.loc[shfl_ids[n_train:],
                            data_a.columns != col_k].values
        y_test = data_a.loc[shfl_ids[n_train:],
                            [col_k]].values.ravel()

        x_synth = data_b.loc[:, data_b.columns != col_k].values
        y_synth = data_b.loc[:, [col_k]].values.ravel()

        if class_method == 'DT':
            #
            # Tree-based model classifier
            #
            try:
                clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
                clf = clf.fit(x_train, y_train)

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'LR':
            #
            # Logistic Regression classifier
            #
            try:
                clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
                clf = clf.fit(x_train, y_train.ravel())

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'MLP':
            #
            # Multilayer Perceptron Classifier
            #
            try:
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(64, 32), random_state=1)
                clf = clf.fit(x_train, y_train.ravel())

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN
        else:
            raise('Unknown classification method')
        accs[col_k] = on_synth / on_holdout

    # if want to return only mean, then use this. If one wants
    # to return crcl for every variable, just return 'accs'
    avg_crcl = {'Avg-CrCl-RS': sum([accs[k] for k in accs.keys()])/len(accs)}

    return avg_crcl, accs

avg_crcl, accs = cross_classification(data_a, data_b, 
                         base_metric='accuracy',
                         class_method='DT')    

import csv

with open('/path_to_mc-medgan/samples/mc-medgan/seerbreast/CrCl-RS.csv', 'w') as f:  
    w = csv.DictWriter(f, accs.keys())
    w.writeheader()
    w.writerow(accs)    

print(avg_crcl)
