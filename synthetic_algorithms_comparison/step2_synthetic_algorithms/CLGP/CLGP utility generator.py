import pandas as pd
import numpy as np
import random
import time
import types
import os
import sys
sys.path.append('..')
from design import Experiment
from collections import defaultdict
from methods.CLGP import CLGP
import tensorflow as tf
from utils import performance_metrics, Logger
from scipy import stats
from scipy.stats import bootstrap
np.random.seed(1234)
from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()
# check if gpu is available
tf.test.is_gpu_available()

# Part I: CLGP algorithm
def clgp_geneator(data, name):
    # set up the clgp parameters
    CLGP_inducing_points = 100
    CLGP_latent_space = 5
    methods = [CLGP(CLGP_inducing_points, CLGP_latent_space, lamb=0.01, reg=10, max_steps=30, name='CLGP')]
    
    name = name
    dataset = enc_data
    methods = methods
    nb_samples = np.inf
    nb_gens = 2
    logger = Logger.Logger()
    
    tf.compat.v1.disable_eager_execution()
    OUTPUT_FOLDER = '../outputs'
    directory = os.path.join(OUTPUT_FOLDER, name)
    # make a new directory with experiment name
    os.makedirs(directory)
    
    # experiment log file will be save in 'directory'
    logger.set_path(directory)
    logger.setup_logger('{}.log'.format(name))
    logger.propagate = False
    logger.info('Experiment directory created.')
    
    nb_samples = nb_samples
    if np.isinf(nb_samples):
        nb_samples = dataset.shape[0]

    for method in methods:
        logger.info('Method {}.'.format(method.name))
        # set method's output directory
        method_directory = os.path.join(directory, method.name)
        # create directory to save method's results/logs
        os.makedirs(method_directory)
        # inform output directory path to the method
        method.set_output_directory(method_directory)
        logger.info('Processing %s' % 'DBDB')
        
        # train model on data
        method.fit(dataset)
        # generate multiple samples from the data generator
        for run in range(nb_gens):
            #logger.info('Sampling synthetic dataset ... ')     
            synth_data = method.generate_samples(nb_samples)
    return synth_data  

## Part II: utility form
def clgp_metric(data_a, data_b):
    metrics = ['kl_divergence',
               'cross_classification',
               'cca_accuracy',
               'cluster_measure',
               'pairwise_correlation_difference',
               'coverage',
               'membership_disclosure',
               'percentage_revealed',
               'attribute_disclosure']
    # get list of available metrics
    metric_func = {a: performance_metrics.__dict__.get(a)
                   for a in dir(performance_metrics)
                   if isinstance(performance_metrics.__dict__.get(a),
                                 types.FunctionType)}
    # initialize results storage structure
    results = {}
    for met in metrics:
        results[met] = list()
    # dict to save performance metrics for the t-th task
    for met in metrics:
        # metric m for method m in the r-th run
        m_m_r = metric_func[met](data_a=data_a, data_b=data_b)
        results[met].append(m_m_r)
        
    result_contents = list()
    method ='CLGP'
    
    # for each metric
    for k in results.keys():
         # for each sampled synthetic dataset
        for i in range(len(results[k])):
            # iterate over metrics for k-th sampled dataset
            type = 'Multiple' if len(results[k][i].keys()) > 1 else 'Single'
            for var in results[k][i].keys():
                result_contents.append([method, k, var, i+1, type, results[k][i][var]])
                # store result_contents list into a dataframe for easier manipulation
                column_names = ['Method', 'Metric', 'Variable', 'Run', 'Type', 'Value']
                df = pd.DataFrame(result_contents, columns=column_names)
    return df


for seed in range(0,1000):
    #real data
    url = '/data/breast_survival_1000samples_'+ str(seed) + '.txt'
    data = pd.read_csv(url, sep=',', )
    data = data.drop(columns=['Unnamed: 0','PatientID'])
    name = 'clgp' + str(seed)
    # synthetic clgp data
    clgp_sub = clgp_geneator(data, name)
    data_a = data
    data_b = clgp_sub
    # generate the metric
    metric_df = clgp_metric(data_a, data_b)
