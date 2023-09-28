###############################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Andre Goncalves <andre@llnl.gov>
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import sys
sys.path.append('..')

import os
import types
import shutil
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from abc import ABCMeta, abstractmethod

from utils import performance_metrics, Logger

OUTPUT_FOLDER = '../outputs'


class Method(object):
    """
    Abstract class to serve as a skeleton for the synthetic generator methods.
    Attributes:
        name = name to be used during the experimental study analysis.
    """

    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Class constructor.
        Args
            name (str): Name of the synthetic generator method.
        """
        self.name = name
        self.logger = Logger.Logger()

    def __str__(self):
        """ Create a string to be used as reference for a class instance. """
        pars = self.get_params()
        if pars is None:
            return '{}'.format(self.__class__.__name__)
        ref = '.'.join(['{}{}'.format(k, pars[k]) for k in pars.keys()])
        ref = ref.replace('_', '').replace('.', '_')
        return '{}_{}'.format(self.__class__.__name__, ref)

    @abstractmethod
    def fit(self):
        """
        Fit model to the data.
        Args
            name (np.array()):
        """
        pass

    @abstractmethod
    def generate_samples(self):
        """
        Generate samples from the trained model.
        Args
            name (pd.DataFrame):
        Return
            name (pd.DataFrame):
        """
        pass

    @abstractmethod
    def set_params(self):
        """
        Set method's parameters.
        Args
            name (np.array()):
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Return defined method's parameters.
        Args
            name (np.array()):
        """
        pass

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())


class Dataset(object):
    """ Abstract class for datasets class. """
    __metaclass__ = ABCMeta

    def __init__(self):
        """ Class initializations. """
        pass

    @abstractmethod
    def prepare_data(self):
        """ Execute data preparation in general such as pre-processing
        data from existing dataset and generate artificial data. """
        pass

    def get_data(self):
        if self.raw_data is None:
            raise ValueError('Must call \'prepare_data\' first.')
        else:
            return self.enc_data

    def decode_data(self, data):
        # Map data back to original categories
        data.apply(lambda x: self.encoding_dict[x.name].inverse_transform(x))
        return data      

class Experiment(object):
    """ Prepares and runs synthetic data generation experiment. It receives
    a dataset (Dataset's class instance), list of methods, list of metrics,
    and the number of samples one wants to generate. It runs all methods
    on the dataset and produce comparison results for all the tested methods
    in terms of the metrics passed by the user.
    """
    def __init__(self, name, dataset, methods, metrics, nb_samples=np.inf, nb_gens=1, decodeflag=1):

        # make sure all inputs have expected values and types
        assert isinstance(name, str), "Experiment name should be a string."
        assert isinstance(dataset, Dataset), "Dataset must be of 'Dataset' type."

        # make sure it received a list of methods
        if not isinstance(methods, list):
            methods = list(methods)
        assert len(methods) > 0, "At least one method must be specified."

        # make sure it received a list of metrics
        if not isinstance(metrics, list):
            metrics = list(metrics)
        assert len(metrics) > 0, "At least one metrics must be specified."

        # check if all methods are valid (instance of Method class)
        for method in methods:
            assert isinstance(method, Method), "Method must be of 'Method' type."

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]

        # check if all metrics are valid (exist in performance_metrics module)
        for metric in metrics:
            assert metric in existing_metrics, '{} is not available is metrics library.'.format(metric)

        # number of samples has to be larger than 0
        assert nb_samples > 0, "Number of samples must be > 0."

        # number of runs has to be larger then 0
        assert nb_gens > 0, "Number of runs must be > 0."

        self.name = name
        self.dataset = dataset
        self.methods = methods
        self.metrics = metrics
        self.nb_samples = nb_samples
        self.nb_gens = nb_gens
	self.decodeflag = decodeflag
        self.logger = Logger.Logger()

    def execute(self):
        # set experiment output directory
        directory = os.path.join(OUTPUT_FOLDER, self.name)
        # if directory already exists, then delete it
        if os.path.exists(directory):
            shutil.rmtree(directory)
        # make a new directory with experiment name
        os.makedirs(directory)

        # experiment log file will be save in 'directory'
        self.logger.set_path(directory)
        self.logger.setup_logger('{}.log'.format(self.name))
        self.logger.propagate = False
        self.logger.info('Experiment directory created.')

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        nb_samples = self.nb_samples
        if np.isinf(self.nb_samples):
            nb_samples = self.dataset.get_data().shape[0]

        # execute all methods passed through 'methods' attribute
        for method in self.methods:

            self.logger.info('Method {}.'.format(method.name))

            # set method's output directory
            method_directory = directory


            # inform output directory path to the method
            method.set_output_directory(method_directory)

            self.logger.info('Processing %s' % self.dataset.db_name)

            # train model on data
            method.fit(self.dataset.get_data())
            
            # save model to file
            output_fname = os.path.join(method_directory,
                                        'method_{}.pkl'.format(method.__str__()))
            with open(output_fname, 'wb') as fh:
                print("Pickling ",output_fname)
                pickle.dump(method, fh)            

            # initialize results storage structure
            results = {}
            for met in self.metrics:
                results[met] = list()
            
            # generate multiple samples from the data generator
            for run in range(self.nb_gens):
                self.logger.info('Sampling synthetic dataset ... ')
		if self.decodeflag == 1:
                	synth_data = self.dataset.decode_data(method.generate_samples(nb_samples))
                # save synthetic data to csv file
                output_fname = os.path.join(method_directory,
                                        'breast_survival_1000samples_{}-sample.csv'.format(run.__str__()))
                synth_data.to_csv(output_fname)
                else:
			synth_data = method.generate_samples(nb_samples)
                # save synthetic data to csv file
                output_fname = os.path.join(method_directory,
                                        'breast_survival_1000samples_{}-sample.csv'.format(run.__str__()))
                self.dataset.decode_data(synth_data).to_csv(output_fname)
                
                # dict to save performance metrics for the t-th task
                for met in self.metrics:
                    # metric m for method m in the r-th run
		    if self.decodeflag == 1:
                    	m_m_r = metric_func[met](data_a=self.dataset.raw_data, data_b=synth_data)
		    else:
                    	m_m_r = metric_func[met](data_a=self.dataset.get_data(), data_b=synth_data)
                    results[met].append(m_m_r)

            # save results to file
            output_fname = os.path.join(method_directory,
                                        'results_{}.pkl'.format(method.__str__()))
            with open(output_fname, 'wb') as fh:
                pickle.dump(results, fh)
            self.logger.info('Results stored in %s' % (output_fname))
            
            result_contents = list()
            for l in range(len(results['percentage_revealed'])):
                result_contents.append([method.name, 'percentage_revealed', '-', l+1, 'Single', results['percentage_revealed'][l]])
            del results['percentage_revealed']
            # for each metric
            for k in results.keys():
                # for each sampled synthetic dataset
                for i in range(len(results[k])):
                    # iterate over metrics for k-th sampled dataset
                    type = 'Multiple' if len(results[k][i].keys()) > 1 else 'Single'
                    for var in results[k][i].keys():
                        result_contents.append([method.name, k, var, i+1, type, results[k][i][var]])

            # store result_contents list into a dataframe for easier manipulation
            column_names = ['Method', 'Metric', 'Variable', 'Run', 'Type', 'Value']
            df = pd.DataFrame(result_contents, columns=column_names)
            df.to_csv(os.path.join(method_directory, 'breast_survival_1000samples_{}-utility_metrics.csv'.format(run.__str__())))


