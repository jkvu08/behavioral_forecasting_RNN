# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:23:29 2021

Use Bayesian hyperparameter optimization to select vanilla RNN and encoder-decoder RNN hyperparameters and lookback period.
Objective of the model is to predict the multiclass (4) behavior of wild lemurs based on the lemurs' prior behaviors, intrinsic and extrinsic conditions.

@author: Jannet
"""
import numpy as np
from numpy import random, newaxis
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from pandas import read_csv, DataFrame, concat, Series, merge
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
import os, time, itertools, random, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from hyperopt.pyll.base import scope
from keras.callbacks import Callback
import keras.backend as K
from hyperopt.early_stop import no_progress_loss
#import threading
#from threading import Thread
#import concurrent.futures
#from numba import cuda
import pickle
import joblib
import seaborn as sns
import ray
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, Conv1D, Activation, RepeatVector, TimeDistributed, Flatten, MaxPooling1D, ConvLSTM2D
from tensorflow.keras.metrics import CategoricalAccuracy, Accuracy, Precision, Recall, AUC
#from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow_addons.metrics import F1Score
#from tensorflow.keras.utils import to_categorical
# import CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import keras.backend as K
from tensorflow.compat.v1 import ConfigProto, InteractiveSession, Session

######################
#### Data Import #####
######################
# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

# import packages from file
import behavior_model_func as bmf

# import training datafiles
datasub =  read_csv('kian_trainset_focal.csv', 
                    header = 0, 
                    index_col = 0)

# get rid of these predictors, sparse or irrelavant
datasub = datasub.drop(columns=['since_social','flower_shannon','fruit_shannon'], 
                       axis = 1) 

datasub.columns
# Index(['ID', 'TID', 'track_position', 'track_length', 'focal', 'year',
#        'individual_continuity', 'feed', 'rest', 'social', 'travel',
#        'gestation', 'lactation', 'mating', 'nonreproductive', 'sex',
#        'fragment', 'length', 'position', 'since_rest', 'since_feed',
#        'since_travel', 'adults', 'infants', 'juveniles', 'rain', 'temperature',
#        'flower_count', 'fruit_count', 'years', 'minutes_sin', 'minutes_cos',
#        'doy_sin', 'doy_cos', 'fcode'],
#       dtype='object')

# subset and reorder predictors
datasub = datasub[['ID', 'TID', 'track_position', 'track_length', 'focal', 'year', # identifiers
                   'feed', 'rest', 'social', 'travel', # behaviors         
                   'individual_continuity', 'length', 'position', # sampling features 
                   'gestation', 'lactation', 'mating', 'nonreproductive', 'sex', # internal features 
                   'since_rest', 'since_feed', 'since_travel', # internal features
                   'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count', # external features
                   'years', 'minutes_sin', 'minutes_cos', 'doy_sin','doy_cos', # external/time features
                   'adults', 'infants', 'juveniles']] # external/group features

#####################################
#### Hyperoptimization functions ####
#####################################
def hyp_nest(params, features, targets):
    '''
    construct vanilla RNN or encoder-decode RNN based on parameter dictionary specifications

    Parameters
    ----------
    params : dict, dictionary of paramters and hyperparameters
    features : int, number of features used for prediction
    targets : int, number of targets (classes) predicted

    Raises
    ------
    Exception
        something other than 'VRNN' designating vanilla RNN or 'ENDE' designating encoder-decoder RNN was specified in params['atype']

    Returns
    -------
    model : RNN model

    '''
    if params['atype'] == 'VRNN':
        model = bmf.build_rnn(features, 
                              targets, 
                              lookback = params['lookback'], 
                              neurons_n = params['neurons_n'],
                              hidden_n = [params['hidden_n0'],params['hidden_n1']],
                              learning_rate =params['learning_rate'],
                              dropout_rate = params['dropout_rate'],
                              layers = params['hidden_layers'], 
                              mtype = params['mtype'], 
                              cat_loss = params['loss'])
    elif params['atype'] == 'ENDE':
        model = bmf.build_ende(features, 
                               targets, 
                               lookback = params['lookback'], 
                               n_outputs = params['n_outputs'], 
                               neurons_n0 = params['neurons_n0'],
                               neurons_n1 = params['neurons_n1'],
                               hidden_n = [params['hidden_n0'],params['hidden_n1']],
                               td_neurons = params['td_neurons'], 
                               learning_rate =params['learning_rate'],
                               dropout_rate = params['dropout_rate'],
                               layers = params['hidden_layers'], 
                               mtype = params['mtype'],
                               cat_loss = params['loss'])
    else:
        raise Exception ('invalid model architecture')    
    return model

def hyperoptimizer_rnn(params):
    """
    hyperparameter optimizer objective function to be used with hyperopt

    Parameters
    ----------
    params : hyperparameter search space
        
    Raises
    ------
    Exception
        something other than 'full','behavior','internal','external' designated in params['predictor']

    Returns
    -------
    obj_dict: dictionary 
        {loss: loss value to optimize through minimization (i.e., validation loss)
         status: default value for hyperopt
         params: the hyperparameter values being tested
         train_loss: training loss
         train_f1: training f1 score
         train_acc: training accuracy
         val_loss: validation loss
         val_f1: validation f1 score
         val_acc: validation accuracy}
    """
    targets = 4 # set number of targets (4 behavior classes)
    train, test = bmf.split_dataset(datasub, 2015) # split the data by year
    # format the training and testing data for the model
    train_X, train_y, train_dft = bmf.to_supervised(data = train.iloc[:,6:34], 
                                                    TID = train['TID'], 
                                                    window = 1, 
                                                    lookback = params['lookback'], 
                                                    n_output = params['n_outputs']) 
    # format testing data
    test_X, test_y, test_dft = bmf.to_supervised(data = test.iloc[:,6:34], 
                                                 TID = test['TID'],
                                                 window = 1, 
                                                 lookback = params['lookback'], 
                                                 n_output = params['n_outputs'])
    
    # if encoder-decode model and predict 1 timestep, reconfigure 2d y to 3d
    if params['atype'] == 'ENDE' and params['n_outputs'] == 1:
        test_y = test_y[:,newaxis,:]
        train_y = train_y[:,newaxis,:]
    
    # assign and format feature set
    if params['predictor'] == 'full': # use full set of features
        features=28 # set feature number
    elif params['predictor'] == 'behavior': # use only prior behaviors as features
        features=4 # set features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:4]) 
        test_X = np.copy(test_X[:,:,0:4])
    elif params['predictor'] == 'internal': # use internal features (behaviors and sex) as features
        features=12 # set features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:12]) 
        test_X = np.copy(test_X[:,:,0:12])
    elif params['predictor'] == 'external': # use the extrinsic conditions
        features = 17
        # subset only extrinsic features
        train_X = np.copy(train_X[:,:,8:25])
        test_X = np.copy(test_X[:,:,8:25])  
    else:
        raise Exception ('invalid feature selection')   
    model = hyp_nest(params, features, targets) # build model
    # fit model and extract evaluation epochs, loss and metrics
    _, avg_eval = bmf.eval_iter(model, 
                                params, 
                                train_X, 
                                train_y, 
                                test_X, 
                                test_y, 
                                patience = params['patience'], 
                                max_epochs = params['max_epochs'], 
                                atype = params['atype'], 
                                n = params['iters']) 
    # reassign epoch based on early stopping
    params['epochs'] = int(avg_eval['epochs'])
    # convert evaluation loss and metrics to dictionary
    avg_dict = avg_eval[1:].to_dict() # don't need average epochs ran (so get rid of first entry)
    obj_dict = {'loss': avg_eval['val_loss'], # use validation loss as loss objective
                'status': STATUS_OK,  # set as default for hyperopt
                'params': params} # output parameters
    obj_dict.update(avg_dict)
    print('Best validation for trial:', obj_dict['val_f1']) # print the validation score
    return obj_dict

def run_trials(filename, objective, space, rstate, initial = 20, trials_step = 1):
    """
    Run and save hyperoptimization experiments.
    Used to run trials in small batches and periodically save to file.
    
    Parameters
    ----------
    filename : str, trial filename
    objective : function, objective
    space : dict, parameter search space 
    rstate: int, set random state for consistency across trials
    initial: int, initial number of trials, should be > 20 
    trials_steps: int, how many additional trials to do after loading saved trials.
    
    Returns
    -------
    max_trials: int, number of experiments ran

    """
    max_trials = initial  # set the initial trials to run (should be at least 20, since hyperopt selects parameters randomly for the first 20 trials)
    try:  # try to load an already saved trials object, and increase the max
        trials = joblib.load(filename) # load file 
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step # increase the max_evals value
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # if trial file cannot be found
        trials = Trials() # create a new trials object
    # run the search
    best = fmin(fn=objective, # objective function to minimize
                space=space, # parameter space to search 
                algo=tpe.suggest, # search algorithm, use sequential search
                max_evals=max_trials, # number of maximum trials
                trials=trials, # previously run trials
                rstate = np.random.default_rng(rstate)) # seed
    print("Best:", best)
    print("max_evals:", max_trials)
    joblib.dump(trials, filename) # save the trials object
    return max_trials

def iter_trials(space, seed, steps, n = 1000):
    """
    function to run hyperparameter optimization experiments    

    Parameters
    ----------
    space : dict, parameter search space
    objective : func, objective function
    seed : int, seed 
    steps : int, number of experiments to run before saving
    n : int, number of experiments

    Returns
    -------
    None.

    """
    g = 0 # initial counter
    while g < n: # which number of experiments is less than maximum experiments
    # run the experiment
        g = run_trials(filename =  space['atype'] + '_' +space['mtype'] +'_' + space['predictor'] + '_' + str(seed) +'.pkl', # assign filename
                       objective = hyperoptimizer_rnn, # assign objective function for hyperopt evaluation
                       space = space, # search space
                       rstate = seed, # random seed
                       initial= 25, # number of experiments to run before first save, want at least 20 since hyperopt uses random parameters within search space for the first 20 experiments
                       trials_step = steps) # number of experiments to run thereafter before saving
  
############################
#### Model optimization ####
############################
# test model construction wrapper for vanilla RNN
# specify parameters/hyperparameters
vrnn_params = {'atype': 'VRNN',
              'mtype': 'GRU',
              'lookback': 5, 
              'hidden_layers': 2,
              'neurons_n': 20,
              'hidden_n0': 10,
              'hidden_n1': 10,
              'learning_rate': 0.001,
              'dropout_rate': 0.3,               
              'loss': False,
              'epochs': 5,
              'batch_size': 512,
              'weights_0': 1,
              'weights_1': 1,
              'weights_2': 3,
              'weights_3': 1}

model = hyp_nest(vrnn_params, 28, 4) # 28 features, 4 behavior classes

model.summary() # looks identical to prior model as expected
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 28)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                3000      
# _________________________________________________________________
# dropout (Dropout)            (None, 20)                0         
# _________________________________________________________________
# dense (Dense)                (None, 10)                210       
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,254
# Trainable params: 3,254
# Non-trainable params: 0
# _________________________________________________________________

# test model construction wrapper for encoder-decoder model
# specify parameters/hyperparameters 
ende_params = {'atype': 'ENDE',
              'mtype': 'GRU',
              'lookback': 5,
              'n_outputs': 1,
              'hidden_layers': 1,
              'neurons_n0': 20,
              'neurons_n1': 20,
              'hidden_n0': 10,
              'hidden_n1': 10,
              'td_neurons': 5,
              'learning_rate': 0.001,
              'dropout_rate': 0.3,               
              'loss': False,
              'epochs': 100,
              'batch_size': 512,
              'weights_0': 1,
              'weights_1': 1,
              'weights_2': 3,
              'weights_3': 1}

model = hyp_nest(ende_params, 28, 4) # 28 features, 4 behavior classes
model.summary() # looks identical to prior model as expected
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 28)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                3000      
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 20)                0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector (RepeatVector) (None, 1, 10)             0         
# _________________________________________________________________
# gru (GRU)                    (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed (TimeDistri (None, 1, 5)              105       
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, 1, 4)              24        
# =================================================================
# Total params: 5,259
# Trainable params: 5,259
# Non-trainable params: 0
# _________________________________________________________________

# run hyperparameter objective function 
# specify set parameters/hyperparameters for vanilla RNN
vrnn_params = {'atype': 'VRNN',
              'mtype': 'GRU',
              'iters': 1,
              'lookback': 5, 
              'n_outputs': 1,
              'predictor':'full',
              'hidden_layers': 1,
              'neurons_n': 20,
              'hidden_n0': 10,
              'hidden_n1': 10,
              'learning_rate': 0.001,
              'dropout_rate': 0.3,               
              'loss': False,
              'epochs': 100,
              'max_epochs': 100,
              'patience': 30,
              'batch_size': 512,
              'weights_0': 1,
              'weights_1': 1,
              'weights_2': 3,
              'weights_3': 1}

vrnn_obj = hyperoptimizer_rnn(vrnn_params) # run objective function for vanilla RNN
# Best validation for trial: 0.42486771941185

vrnn_obj # view objective function output 
# {'loss': 0.5751951932907104,
#  'status': 'ok',
#  'params': {'atype': 'VRNN',
#   'mtype': 'GRU',
#   'iters': 1,
#   'lookback': 5,
#   'n_outputs': 1,
#   'predictor': 'full',
#   'hidden_layers': 1,
#   'neurons_n': 20,
#   'hidden_n0': 10,
#   'hidden_n1': 10,
#   'learning_rate': 0.001,
#   'dropout_rate': 0.3,
#   'loss': False,
#   'epochs': 49.0,
#   'max_epochs': 100,
#   'patience': 30,
#   'batch_size': 512,
#   'weights_0': 1,
#   'weights_1': 1,
#   'weights_2': 3,
#   'weights_3': 1},
#  'train_loss': 0.5988207459449768,
#  'train_f1': 0.4180199205875397,
#  'train_acc': 0.7597236037254333,
#  'val_loss': 0.5751951932907104,
#  'val_f1': 0.42486771941185,
#  'val_acc': 0.8361517190933228}

# specify parameters/hyperparameters for encoder decoder
ende_params = {'atype': 'ENDE',
              'mtype': 'GRU',
              'iters': 1,
              'lookback': 5, 
              'n_outputs': 1,
              'predictor':'full',
              'hidden_layers': 1,
              'neurons_n': 20,
              'hidden_n0': 10,
              'hidden_n1': 10,
              'td_neurons': 5,
              'learning_rate': 0.001,
              'dropout_rate': 0.3,               
              'loss': False,
              'epochs': 5,
              'max_epochs': 100,
              'patience': 30,
              'batch_size': 512,
              'weights_0': 1,
              'weights_1': 1,
              'weights_2': 3,
              'weights_3': 1}

ende_obj = hyperoptimizer_rnn(ende_params) # run objective function for encoder decoder 
# Best validation for trial: 0.42537400126457214

ende_obj # view objective function output
# {'loss': 0.573643684387207,
#  'status': 'ok',
#  'params': {'atype': 'ENDE',
#   'mtype': 'GRU',
#   'iters': 1,
#   'lookback': 5,
#   'n_outputs': 1,
#   'predictor': 'full',
#   'hidden_layers': 1,
#   'neurons_n': 20,
#   'hidden_n0': 10,
#   'hidden_n1': 10,
#   'td_neurons': 5,
#   'learning_rate': 0.001,
#   'dropout_rate': 0.3,
#   'loss': False,
#   'epochs': 68.0,
#   'max_epochs': 100,
#   'patience': 30,
#   'batch_size': 512,
#   'weights_0': 1,
#   'weights_1': 1,
#   'weights_2': 3,
#   'weights_3': 1},
#  'train_loss': 0.59934401512146,
#  'train_f1': 0.41581910848617554,
#  'train_acc': 0.7629953622817993,
#  'val_loss': 0.573643684387207,
#  'val_f1': 0.42537400126457214,
#  'val_acc': 0.8343235850334167}

# specify vanilla RNN parameter space
space_vrnn = {'atype'                  : 'VRNN',
              'mtype'                  : 'GRU',
              'iters'                  : 1,
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'n_outputs'              : 1,
              'predictor'              : 'behavior',
              'hidden_layers'          : scope.int(hp.quniform('layers',0,2,1)),
              'neurons_n'              : scope.int(hp.quniform('neurons_n',5,50,5)),
              'hidden_n0'              : scope.int(hp.quniform('hidden_n0',5,50,5)),
              'hidden_n1'              : scope.int(hp.quniform('hidden_n1',5,50,5)),
              'learning_rate'          : 0.001,
              'dropout_rate'           : hp.quniform('dropout_rate',0.1,0.9,0.1),
              'loss'                   : False,
              'epochs'                 : 50,
              'max_epochs'             : 50,
              'patience'               : 30,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,5,0.5),
              'weights_1'              : 1,
              'weights_2'              : scope.int(hp.quniform('weights_2',1,25,1)),
              'weights_3'              : scope.int(hp.quniform('weights_3',1,10,1))
              }

seed = 123
run_trials(filename =  path + space_vrnn['atype'] + '_' + space_vrnn['mtype'] +'_' + space_vrnn['predictor'] + '_' + str(seed) +'.pkl', # assign filename
                       objective = hyperoptimizer_rnn, # assign objective function for hyperopt evaluation
                       space = space_vrnn, # search space
                       rstate = seed, # random seed
                       initial= 3, # number of experiments to run before first save
                       trials_step = 2) # number of experiments to run thereafter before saving

# output 
# validation for trial:
# 0.4292759597301483                                                             
# 100%|██████████| 3/3 [01:56<00:00, 38.90s/trial, best loss: 0.5712067484855652]
# Best: {'dropout_rate': 0.4, 'hidden_n0': 45.0, 'hidden_n1': 50.0, 'layers': 1.0, 'lookback': 11.0, 'neurons_n': 35.0, 'weights_0': 3.0, 'weights_2': 22.0, 'weights_3': 1.0}
# max_evals: 3
# ran 3 since it was the initial run. 

# run again to see if model is picking up from before
run_trials(filename =  path + space_vrnn['atype'] + '_' + space_vrnn['mtype'] +'_' + space_vrnn['predictor'] + '_' + str(seed) +'.pkl', # assign filename
                       objective = hyperoptimizer_rnn, # assign objective function for hyperopt evaluation
                       space = space_vrnn, # search space
                       rstate = seed, # random seed
                       initial= 3, # number of experiments to run before first save
                       trials_step = 2) # number of experiments to run thereafter before saving

# output
# Best validation for trial:                                                     
# 0.4228237569332123                                                             
# 100%|██████████| 5/5 [00:54<00:00, 27.14s/trial, best loss: 0.5712067484855652]
# Best: {'dropout_rate': 0.4, 'hidden_n0': 45.0, 'hidden_n1': 50.0, 'layers': 1.0, 'lookback': 11.0, 'neurons_n': 35.0, 'weights_0': 3.0, 'weights_2': 22.0, 'weights_3': 1.0}
# max_evals: 5
# see that the runs continued from prior, resulting in 5 total experiments saved

### NEED TO EVALUATE HYPEROPT MODELS VISUALS

# specify encoder-decoder RNN parameter space
space_ende = {'atype'                  : 'ENDE',
              'mtype'                  : 'GRU',
              'iters'                  : 1,
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'n_outputs'              : 1,
              'predictor'              : 'behavior',
              'hidden_layers'          : scope.int(hp.quniform('layers',0,2,1)),
              'neurons_n0'              : scope.int(hp.quniform('neurons_n0',5,50,5)),
              'neurons_n1'              : scope.int(hp.quniform('neurons_n1',5,50,5)),
              'hidden_n0'              : scope.int(hp.quniform('hidden_n0',5,50,5)),
              'hidden_n1'              : scope.int(hp.quniform('hidden_n1',5,50,5)),
              'td_neurons'             : scope.int(hp.quniform('td_neurons',5,50,5)),
              'learning_rate'          : 0.001,
              'dropout_rate'           : hp.quniform('dropout_rate',0.1,0.9,0.1),
              'loss'                   : False,
              'epochs'                 : 50,
              'max_epochs'             : 50,
              'patience'               : 30,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,5,0.5),
              'weights_1'              : 1,
              'weights_2'              : scope.int(hp.quniform('weights_2',1,25,1)),
              'weights_3'              : scope.int(hp.quniform('weights_3',1,10,1))
              }

seed = 123
run_trials(filename =  path + space_ende['atype'] + '_' + space_ende['mtype'] +'_' + space_ende['predictor'] + '_' + str(seed) +'.pkl', # assign filename
                       objective = hyperoptimizer_rnn, # assign objective function for hyperopt evaluation
                       space = space_ende, # search space
                       rstate = seed, # random seed
                       initial= 3, # number of experiments to run before first save
                       trials_step = 2) # number of experiments to run thereafter before saving

# output
# Best validation for trial:                                                     
# 0.3424544930458069                                                             
# 100%|██████████| 3/3 [02:27<00:00, 49.25s/trial, best loss: 0.5815421938896179]
# Best: {'dropout_rate': 0.8, 'hidden_n0': 25.0, 'hidden_n1': 10.0, 'layers': 1.0, 'lookback': 10.0, 'neurons_n0': 25.0, 'neurons_n1': 40.0, 'td_neurons': 5.0, 'weights_0': 5.0, 'weights_2': 4.0, 'weights_3': 5.0}
# max_evals: 3
# Out[81]: 3

# run using iteration function to run n number of trials
current = time.perf_counter() # keep track of time
iter_trials(space = space_vrnn, 
            seed = 321, 
            steps = 3, 
            n = 30)
print('Took ' + str(round((time.perf_counter()-current)/60,2)) + ' mins')

# Output:
# Best validation for trial:                                                       
# 0.42851927876472473                                                              
# 100%|██████████| 31/31 [02:06<00:00, 42.14s/trial, best loss: 0.5679725408554077]
# Best: {'dropout_rate': 0.4, 'hidden_n0': 30.0, 'hidden_n1': 50.0, 'layers': 2.0, 'lookback': 13.0, 'neurons_n': 35.0, 'weights_0': 2.5, 'weights_2': 6.0, 'weights_3': 10.0}
# max_evals: 31
# Took 23.52 mins


# ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
# @ray.remote
# class Simulator(object):
#     def __init__(self,seed):
#         import tensorflow as tf
#         import keras
#         from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, Conv1D, Activation, RepeatVector, TimeDistributed, Flatten, MaxPooling1D, ConvLSTM2D
#         #from tensorflow.keras.preprocessing import sequence
#         from tensorflow.keras.optimizers import Adam
#         from tensorflow.keras.models import Sequential, Model
#         from tensorflow_addons.metrics import F1Score
#         #from tensorflow.keras.utils import to_categorical
#         # import CategoricalAccuracy, CategoricalCrossentropy
#         #from tensorflow.compat.v1.keras.layers import mean_per_class_accuracy
#         from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#         from tensorflow.keras.callbacks import EarlyStopping
#         from keras.callbacks import Callback
#         import keras.backend as K
#         from tensorflow.compat.v1 import ConfigProto, InteractiveSession, Session

#         # num_CPU = 1
#         # num_cores = 7
#         # config = ConfigProto(intra_op_parallelism_threads = num_cores,
#         #                       inter_op_parallelism_threads = num_cores,
#         #                       device_count={'CPU':num_CPU})
        
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
    
#         # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3,allow_growth=True) 
#         # config = ConfigProto(gpu_options)
#         # self.sess = Session(config=config)
#         self.seed = seed

#     def iter_trials_vrnn(self):
#         g = 0
#         while g < 1000:
#             g = run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(self.seed)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =self.seed, initial=25, trials_step=2)
#        # self.sess.close()
        
#     def iter_trials_ende(self):
#         g = 0
#         while g < 1000:
#             g = run_trials(filename = 'ende_vv_trials_seed'+str(self.seed)+'.pkl',objective =hyperoptimizer_ende, space =space_ende, rstate =self.seed, initial=25, trials_step=3)
    
#     def iter_trials_vbonly(self):
#         g = 0
#         while g < 1000:
#             g = run_trials(filename = 'vrnn_bonly_vv_trials_seed'+str(self.seed)+'.pkl',objective =hyperoptimizer_vrnn_bonly, space =space_bonly, rstate =self.seed, initial=25, trials_step=3)

# start = time.perf_counter()
# simulators = [Simulator.remote(a) for a in [123,619,713]]
# results = ray.get([s.iter_trials_vrnn.remote() for s in simulators])
# finish = time.perf_counter()
# print('Took '+str((finish-start)/(3600)) + 'hours')


# 144,302,529
# # load up trials
# ende20 = joblib.load("ende_vv_trials_seed20.pkl")
# ende51 = joblib.load("ende_vv_trials_seed51.pkl")
# ende90 = joblib.load("ende_vv_trials_seed90.pkl")

# vrnn20 = joblib.load("vanilla_rnn_vv_trials_seed20.pkl")
# vrnn16 = joblib.load("vanilla_rnn_vv_trials_seed16.pkl")
# vrnn06 = joblib.load("vanilla_rnn_vv_trials_seed6.pkl")

# seeds = random.sample(range(100000),3)


# current = time.perf_counter()
# run_trials(filename = 'ende' + '_' +space_ende['mtype'] +'_' + space_ende['covariate'] + '_'+str(114)+'.pkl',objective =hyperoptimizer_ende, space =space_ende, rstate =114, initial=2, trials_step=2)
# #run_trials(filename = 'vrnn' + space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seeds[0])+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seeds[0], initial=2, trials_step=2)
# print((time.perf_counter()-current)/60)

# run_trials(filename = 'vrnn' + space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seeds[1])+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seeds[1], initial=25, trials_step=2)
# run_trials(filename = 'vrnn' + space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seeds[2])+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seeds[2], initial=25, trials_step=2)

# run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(312)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =312, initial=2, trials_step=2)
# run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(223)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =223, initial=25, trials_step=2)
# run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(969)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =969, initial=25, trials_step=2)

current = time.perf_counter()
iter_trials_vrnn(random.sample(range(100000),1)[0])
print((time.perf_counter()-current)/60)

current = time.perf_counter()
iter_trials_ende(random.sample(range(100000),1)[0])
print((time.perf_counter()-current)/3600)

current = time.perf_counter()
iter_trials_vrnn(42577)
print((time.perf_counter()-current)/60)

iter_trials_vrnn(523)

current = time.perf_counter()
iter_trials_ende(56924)
print((time.perf_counter()-current)/3600)

