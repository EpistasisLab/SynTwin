#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
sys.path.append('..')
from design import Experiment
from design import Dataset
from SEERData import SEERData
from MixtureProductMultinomials import MixtureProductMultinomials

data_path = "/data/breast_survival_1000samples_" + str(sys.argv[1]-1) + ".txt" #Path to input file
dataset = SEERData(data_path=data_path)
dataset.prepare_dataset()    

# default config per paper - K=30, alpha=10, n_gibbs_steps=10000, burn_in_steps = 1000
K = 30
burn_in_steps = 1000
n_gibbs_steps = 10000
methods = [MixtureProductMultinomials(K=K, burn_in_steps=burn_in_steps, n_gibbs_steps=n_gibbs_steps, name='MPoM')] 

# list of metrics to measure method's performance
metrics = ['kl_divergence',
           'cross_classification',
           'cca_accuracy',
           'cluster_measure',
           'pairwise_correlation_difference',
            'coverage',
           'membership_disclosure',
            'percentage_revealed',
           'attribute_disclosure'
]

# create an experiment and execute it
exp_folder = data_path.split("/")[-1].split(".")[0] + "_" + str(K) + "_" + str(burn_in_steps) + "_" + str(n_gibbs_steps)
out_folder = '/path_to_save/'
exp = Experiment(out_folder, exp_folder, dataset, methods,
                 metrics, nb_gens=1)  #nb_gens - number of runs
exp.execute()
    
