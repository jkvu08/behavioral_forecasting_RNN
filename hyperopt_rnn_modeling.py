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
import hyperopt_vis_func as hvf

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
         eval_iters: model evaluation results for each iteration (i.e., training and validation loss, f1 score and accuracy),
         avg_eval: average model evaluation across iterations (i.e., training and validation loss, f1 score and accuracy)}
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
    eval_df, avg_eval = bmf.eval_iter(model, 
                                      params, 
                                      train_X, 
                                      train_y, 
                                      test_X, 
                                      test_y, 
                                      patience = params['patience'], 
                                      max_epochs = params['max_epochs'], 
                                      atype = params['atype'], 
                                      n = params['iters']) 
    
    obj_dict = {'loss': avg_eval['val_loss'], # use validation loss as loss objective
                'status': STATUS_OK,  # set as default for hyperopt
                'params': params, # output parameters
                'eval_iters': eval_df, # epoch, loss and evaluation for each iteration 
                'avg_eval': avg_eval} # average epoch, loss and evaluation across iterations

    print('Best validation for trial:', avg_eval['val_f1']) # print the validation score
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

def iter_trials(space, path, seed, steps, n = 1000):
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
        g = run_trials(filename =  path + space['atype'] + '_' +space['mtype'] +'_' + space['predictor'] + '_' + str(seed) +'.pkl', # assign filename
                       objective = hyperoptimizer_rnn, # assign objective function for hyperopt evaluation
                       space = space, # search space
                       rstate = seed, # random seed
                       initial= 25, # number of experiments to run before first save, want at least 20 since hyperopt uses random parameters within search space for the first 20 experiments
                       trials_step = steps) # number of experiments to run thereafter before saving
  
####################
#### Model runs ####
####################
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
              'max_epochs': 100,
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
              'max_epochs': 100,
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
              'iters': 5,
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
              'max_epochs': 10,
              'patience': 5,
              'batch_size': 512,
              'weights_0': 1,
              'weights_1': 1,
              'weights_2': 3,
              'weights_3': 1}

vrnn_obj = hyperoptimizer_rnn(vrnn_params) # run objective function for vanilla RNN
# Best validation for trial: 0.4163888990879059

vrnn_obj # view objective function output 
# {'loss': 0.5822282552719116,
#  'status': 'ok',
#  'params': {'atype': 'VRNN',
#   'mtype': 'GRU',
#   'iters': 5,
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
#   'max_epochs': 10,
#   'patience': 5,
#   'batch_size': 512,
#   'weights_0': 1,
#   'weights_1': 1,
#   'weights_2': 3,
#   'weights_3': 1},
#   'eval_iters':      epochs  iter  train_loss  train_f1  train_acc  val_loss    val_f1    val_acc
#               0      10      0     0.616888    0.399122  0.766169   0.587300    0.411824  0.839637
#               1      10      1     0.610635    0.401822  0.768684   0.587308    0.409688  0.840665
#               2      10      2     0.608055    0.402600  0.761677   0.584296    0.410514  0.841122
#               3      10      3     0.602933    0.414899  0.759309   0.576787    0.424864  0.837294
#               4      10      4     0.600705    0.418485  0.759479   0.575450    0.425055  0.835980,
#  'avg_eval': epochs        10.000000
#              train_loss     0.607843
#              train_f1       0.407386
#              train_acc      0.763064
#              val_loss       0.582228
#              val_f1         0.416389
#              val_acc        0.838940
#              dtype: float64}

# specify parameters/hyperparameters for encoder decoder
ende_params = {'atype': 'ENDE',
              'mtype': 'GRU',
              'iters': 5,
              'lookback': 5, 
              'n_outputs': 1,
              'predictor':'full',
              'hidden_layers': 1,
              'neurons_n0': 20,
              'neurons_n1': 20,
              'hidden_n0': 10,
              'hidden_n1': 10,
              'td_neurons': 5,
              'learning_rate': 0.001,
              'dropout_rate': 0.3,               
              'loss': False,
              'max_epochs': 10,
              'patience': 5,
              'batch_size': 512,
              'weights_0': 1,
              'weights_1': 1,
              'weights_2': 3,
              'weights_3': 1}

ende_obj = hyperoptimizer_rnn(ende_params) # run objective function for encoder decoder 
# Best validation for trial: 0.4121915400028229

ende_obj # view objective function output
# {'loss': 0.5839401364326477,
#  'status': 'ok',
#  'params': {'atype': 'ENDE',
#   'mtype': 'GRU',
#   'iters': 5,
#   'lookback': 5,
#   'n_outputs': 1,
#   'predictor': 'full',
#   'hidden_layers': 1,
#   'neurons_n0': 20,
#   'neurons_n1': 20,
#   'hidden_n0': 10,
#   'hidden_n1': 10,
#   'td_neurons': 5,
#   'learning_rate': 0.001,
#   'dropout_rate': 0.3,
#   'loss': False,
#   'max_epochs': 10,
#   'patience': 5,
#   'batch_size': 512,
#   'weights_0': 1,
#   'weights_1': 1,
#   'weights_2': 3,
#   'weights_3': 1},
#  'eval_iters':    epochs  iter  train_loss  train_f1  train_acc  val_loss    val_f1   val_acc
#                0      10     0    0.614793  0.401715   0.777376  0.586817  0.412148  0.845293
#                1      10     1    0.610315  0.404417   0.776888  0.586387  0.411323  0.843236
#                2      10     2    0.608774  0.404630   0.775008  0.585247  0.413813  0.840951
#                3      10     3    0.607342  0.409331   0.775814  0.583371  0.414612  0.844093
#                4      10     4    0.604604  0.412848   0.773299  0.577879  0.421203  0.840894,
#  'avg_eval': epochs        10.000000
#              train_loss     0.609165
#              train_f1       0.406588
#              train_acc      0.775677
#              val_loss       0.583940
#              val_f1         0.414620
#              val_acc        0.842893
#              dtype: float64}

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
                       trials_step = 5) # number of experiments to run thereafter before saving

# output of 3 hyperopt experiments
# validation for trial:
# 0.4292759597301483                                                             
# 100%|██████████| 3/3 [01:56<00:00, 38.90s/trial, best loss: 0.5712067484855652]
# Best: {'dropout_rate': 0.4, 'hidden_n0': 45.0, 'hidden_n1': 50.0, 'layers': 1.0, 'lookback': 11.0, 'neurons_n': 35.0, 'weights_0': 3.0, 'weights_2': 22.0, 'weights_3': 1.0}
# max_evals: 3 

# run again to make sure model is picking up from before
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

# examine hyperopt experiment results
# load the hyperopt experiments
trials = joblib.load(path + 'VRNN_GRU_behavior_123.pkl')
# convert trial results into dataframe
trial_df = hvf.trials_to_df(trials)

trial_df.columns # examine dataframe columns to get sense of data structure
# Index(['atype', 'batch_size', 'dropout_rate', 'hidden_layers', 'hidden_n0',
#        'hidden_n1', 'iters', 'learning_rate', 'lookback', 'loss', 'max_epochs',
#        'mtype', 'n_outputs', 'neurons_n', 'patience', 'predictor', 'weights_0',
#        'weights_1', 'weights_2', 'weights_3', 'epochs', 'train_loss',
#        'train_f1', 'train_acc', 'val_loss', 'val_f1', 'val_acc'],
#       dtype='object')

# examine the results of the last run of the hyperopt experiments  
# atype                VRNN
# batch_size            512
# dropout_rate          0.7
# hidden_layers           0
# hidden_n0               5
# hidden_n1              40
# iters                   1
# learning_rate       0.001
# lookback               19
# loss                False
# max_epochs             50
# mtype                 GRU
# n_outputs               1
# neurons_n              30
# patience               30
# predictor        behavior
# weights_0             5.0
# weights_1               1
# weights_2              12
# weights_3               1
# epochs               50.0
# train_loss       0.896169
# train_f1         0.434498
# train_acc        0.793011
# val_loss         0.585248
# val_f1            0.42036
# val_acc          0.865653
# Name: 4, dtype: object

# visualize experiment results sequentially
# list of metrics to visualize
hyp_metrics = ['train_loss','val_loss',
               'train_f1','val_acc',
               'train_f1','val_acc',
               'epochs','lookback','dropout_rate', 'neurons_n',
               'hidden_layers','hidden_n0','hidden_n1', 
               'weights_0','weights_2','weights_3']

# plot progression of the hyperparameter search
progress_fig = hvf.hyperopt_progress(trial_df, hyp_metrics)
progress_fig.savefig(path+'vrnn_123_5runs_progress.jpg', dpi = 150)

# plot the training and validation loss and performance metrics against each other
train_val_plots = hvf.train_val_comp(trial_df)
train_val_plots.savefig(path+'vrnn_123_5runs_train_val_plot.jpg', dpi = 150)

# plot kernal density of the hyperparameters against the validation loss
kde_fig = hvf.kdeplots(trial_df,hyp_metrics[7:])
kde_fig.savefig(path+'vrnn_123_5runs_kdeplots.jpg', dpi = 150)

# plot correlation between hyperparameters and metrics
mcorr = trial_df.loc[:, hyp_metrics].corr() # calculate the correlation between hyperparameters and metrics
sns.heatmap(mcorr, 
            xticklabels=mcorr.columns,
            yticklabels=mcorr.columns,
            cmap = 'vlag') 

# use wrapper to generate dataframe of the hyperparameter trials, progression figure, 
# train v. val performance figures, kernel density plot of hyperparameter v. validation loss 
# and correlation plot of metrics
trial_df = hvf.trial_correg_pdf(path = path,
                                filename = 'VRNN_GRU_behavior_123',
                                params = ['epochs','lookback','dropout_rate', 
                                          'neurons_n',
                                          'hidden_layers','hidden_n0','hidden_n1', 
                                          'weights_0','weights_2','weights_3'], 
                                monitor = ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc'])

# run n number of trials using iteration function
current = time.perf_counter() # keep track of time
iter_trials(space = space_vrnn, 
            path = path,
            seed = 123, 
            steps = 5, 
            n = 30)
print('Took ' + str(round((time.perf_counter()-current)/60,2)) + ' mins')

# Best validation for trial:                                                        
# 0.4335796535015106                                                                
# 100%|██████████| 30/30 [07:46<00:00, 93.39s/trial, best loss: 0.5664700865745544] 
# Best: {'dropout_rate': 0.6000000000000001, 'hidden_n0': 20.0, 'hidden_n1': 50.0, 'layers': 0.0, 'lookback': 22.0, 'neurons_n0': 35.0, 'neurons_n1': 35.0, 'td_neurons': 35.0, 'weights_0': 3.5, 'weights_2': 1.0, 'weights_3': 1.0}
# max_evals: 30
# Took 34.84 mins

current = time.perf_counter() # keep track of time
iter_trials(space = space_vrnn, 
            path = path,
            seed = 456, 
            steps = 5, 
            n = 30)
print('Took ' + str(round((time.perf_counter()-current)/60,2)) + ' mins')

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
              'max_epochs'             : 50,
              'patience'               : 30,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,5,0.5),
              'weights_1'              : 1,
              'weights_2'              : scope.int(hp.quniform('weights_2',1,25,1)),
              'weights_3'              : scope.int(hp.quniform('weights_3',1,10,1))
              }

# run using iteration function to run n number of trials
current = time.perf_counter() # keep track of time
iter_trials(space = space_ende, 
            path = path,
            seed = 123, 
            steps = 5, 
            n = 30)
print('Took ' + str(round((time.perf_counter()-current)/60,2)) + ' mins')

# Best validation for trial:                                                       
# 0.42666730284690857                                                              
# 100%|██████████| 30/30 [06:51<00:00, 82.31s/trial, best loss: 0.5675432682037354]
# Best: {'dropout_rate': 0.4, 'hidden_n0': 35.0, 'hidden_n1': 30.0, 'layers': 0.0, 'lookback': 13.0, 'neurons_n0': 20.0, 'neurons_n1': 15.0, 'td_neurons': 25.0, 'weights_0': 4.5, 'weights_2': 9.0, 'weights_3': 5.0}
# max_evals: 30
# Took 32.51 mins

# run using iteration function to run n number of trials
current = time.perf_counter() # keep track of time
iter_trials(space = space_ende, 
            path = path,
            seed = 456, 
            steps = 5, 
            n = 30)
print('Took ' + str(round((time.perf_counter()-current)/60,2)) + ' mins')

# Best validation for trial:                                                        
# 0.4335796535015106                                                                
# 100%|██████████| 30/30 [07:46<00:00, 93.39s/trial, best loss: 0.5664700865745544] 
# Best: {'dropout_rate': 0.6000000000000001, 'hidden_n0': 20.0, 'hidden_n1': 50.0, 'layers': 0.0, 'lookback': 22.0, 'neurons_n0': 35.0, 'neurons_n1': 35.0, 'td_neurons': 35.0, 'weights_0': 3.5, 'weights_2': 1.0, 'weights_3': 1.0}
# max_evals: 30
# Took 34.84 mins

current = time.perf_counter() # keep track of time
iter_trials(space = space_vrnn, 
            path = path,
            seed = 456, 
            steps = 5, 
            n = 25)
print('Took ' + str(round((time.perf_counter()-current)/60,2)) + ' mins')

########################################
#### Model evaluation and selection ####
########################################
# hyperparamters that are being optimized
vrnn_params = ['epochs','lookback', 'neurons_n',
               'hidden_layers','hidden_n0','hidden_n1', 
               'dropout_rate','weights_0','weights_2','weights_3']

selection_metrics = ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc']

# output trial df and visualize results of the two runs
vrnn_123 = hvf.trial_correg_pdf(path = path,
                                 filename = 'VRNN_GRU_behavior_123',
                                 params = vrnn_params, 
                                 monitor = selection_metrics)

vrnn_456 = hvf.trial_correg_pdf(path = path,
                                 filename = 'VRNN_GRU_behavior_456',
                                 params = vrnn_params, 
                                 monitor = selection_metrics)

# get parameter summary
hvf.hypoutput(path = path,
              modelname = 'VRNN_GRU_behavior', 
              ci = 0.90,
              params = vrnn_params,
              burnin = 0,
              maxval = 30)

#               VRNN_GRU_behavior
# train_loss     0.71 (0.73,1.37)
# val_loss       0.57 (0.57,0.59)
# train_f1       0.42 (0.33,0.44)
# val_f1         0.42 (0.34,0.43)
# train_acc       0.77 (0.71,0.8)
# val_acc        0.85 (0.83,0.88)
# epochs               50 (50,50)
# lookback              11 (1,21)
# dropout_rate      0.5 (0.1,0.8)
# neurons_n             30 (5,50)
# hidden_layers           1 (0,2)
# hidden_n0      35.0 (10.0,50.0)
# hidden_n1      45.0 (50.0,50.0)
# weights_0        3.25 (1.0,5.0)
# weights_2             12 (2,24)
# weights_3              6 (1,10)


vrnn_exp, vrnn_combined, vrnn_summary = hvf.convergence_sum(path = path,
                                                            modelname = 'VRNN_GRU_behavior',
                                                            params = selection_metrics + vrnn_params,
                                                            burnin = 0, 
                                                            maxval = 30)

hvf.chain_plots(vrnn_exp, 'dropout_rate')
hvf.trial_chains_output(path = path,
                        modelname = 'VRNN_GRU_behavior',
                        params = vrnn_params,
                        burnin = 0, 
                        maxval = 30)


# hyperparamters that being optimized
ende_params = ['epochs','lookback','dropout_rate', 
               'neurons_n0','neurons_n1','td_neurons',
               'hidden_layers','hidden_n0','hidden_n1', 
               'weights_0','weights_2','weights_3']

# output trial df and visualize results of the two runs
ende_123 = hvf.trial_correg_pdf(path = path,
                                 filename = 'ENDE_GRU_behavior_123',
                                 params = ende_params, 
                                 monitor = selection_metrics)

ende_456 = hvf.trial_correg_pdf(path = path,
                                 filename = 'ENDE_GRU_behavior_456',
                                 params = ende_params, 
                                 monitor = selection_metrics)

ende_exp, ende_combined, ende_summary = hvf.convergence_sum(path = path,
                                                            modelname = 'ENDE_GRU_behavior',
                                                            params = ende_params,
                                                            burnin = 0, 
                                                            maxval = 30)


# compare models
modelnames= ['VRNN_GRU_behavior', 'ENDE_GRU_behavior']

# create list to populate 
rnn_list = []
# run the hypoutput for all model types
for mn in modelnames: 
    if mn[0:4] == 'VRNN':
        entry = hvf.hypoutput(path = path,
                              modelname = mn, 
                              ci = 0.90,
                              params = vrnn_params,
                              burnin = 5,
                              maxval = 30)
    else:
        entry = hvf.hypoutput(path = path,
                              modelname = mn, 
                              ci = 0.90,
                              params = ende_params,
                              burnin = 5,
                              maxval = 30)
    rnn_list.append(entry) 

modelcomp_df = pd.concat(rnn_list, axis = 1, join ='outer')
modelcomp_df.to_csv(path + 'modelcomparison_30runs_ci90.csv')

################################
#### Predictive performance ####
################################
# get summary statistics for each model type
# concatenate experiments for a particular model together
vrnn_df = pd.concat([vrnn_123,vrnn_456], axis=0) 
ende_df = pd.concat([ende_123,ende_456], axis=0) 

# calculate summary functions
vrnn_behavior_sum = hvf.sum_function(df = vrnn_df[selection_metrics], 
                                     filename = 'vrnn_behavior_30runs', 
                                     path = path)

# view data to get sense of output
vrnn_behavior_sum.columns
# Index(['model', 'mean', 'sd', 'median', 'mad', 'lci95', 'uci95', 'lci90',
#        'uci90', 'lci80', 'uci80', 'lci50', 'uci50'],
#       dtype='object')

vrnn_behavior_sum.iloc[0,:]
# model     vrnn_behavior_30runs
# mean                  1.031898
# sd                     0.19613
# median                0.997207
# mad                   0.164722
# lci95                 0.728355
# uci95                 1.430349
# lci90                 0.772152
# uci90                 1.369691
# lci80                 0.796362
# uci80                 1.345614
# lci50                 0.869758
# uci50                  1.19639
# Name: train_loss, dtype: object

# calculate summary functions
ende_behavior_sum = hvf.sum_function(ende_df[selection_metrics], 'ende_behavior_30runs', path)

# view data to get sense of output
# ende_behavior_sum.iloc[0,:]
# model     ende_behavior_30runs
# mean                  0.964753
# sd                    0.193616
# median                0.957458
# mad                   0.154591
# lci95                  0.67004
# uci95                 1.346113
# lci90                 0.673511
# uci90                  1.28895
# lci80                 0.704848
# uci80                 1.242567
# lci50                 0.825524
# uci50                  1.11884
# Name: train_loss, dtype: object

# put cross model comparisons into single file 
cross_comp = pd.concat([vrnn_behavior_sum,ende_behavior_sum],axis = 0)

# modelnames= ['vrnn_f1_GRU_behavior', 'vrnn_f1_GRU_full','vrnn_f1_GRU_extrinsic',
#             'vrnn_f1_LSTM_behavior', 'vrnn_f1_LSTM_full','vrnn_f1_LSTM_extrinsic',
#             'ende_f1_GRU_behavior', 'ende_f1_GRU_full','ende_f1_GRU_extrinsic',
#             'ende_f1_LSTM_behavior', 'ende_f1_LSTM_full','ende_f1_LSTM_extrinsic',
#             ]

#markov_df = read_csv('model_comparison_behavior21_markov.csv', header = 0, index_col = 0)
#actdist_df = read_csv('model_comparison_behavior21_actdist.csv', header = 0, index_col = 0)

# a = sum_function(markov_df.iloc[:,2:], 'markov21_behavior_compcheck')
# b = sum_function(actdist_df.iloc[:,2:], 'actdist21_behavior_compcheck')
# df_tabs = [markov_df, actdist_df]
# df_tabs = [act_dist21, act_dist23, markov21, markov23, rnn_median21, rnn_median21_binom, rnn_best23, rnn_best23_binom]
# filename = ['markov21_check','act_dist21_check']

# filename = ['act_dist21', 'act_dist23', 'markov21', 'markov23', 'rnn_median21', 'rnn_median21_binom', 'rnn_best23', 'rnn_best23_binom']
# empty_list = []
# for i in range(len(df_tabs)):
#     a = sum_function(df_tabs[i].iloc[:,2:], filename[i])
#     empty_list.append(a)

# cross_comp_tab = pd.concat(empty_list)    
# cross_comp_tab.to_csv('cross_comparison_behavior_ci_markov_act_21.csv')
# trials = []
# for file in glob.glob('ende_GRU_extrinsic_56924' +'*.pkl'): # for files with this prefix
#     print(file)    
#     file = joblib.load(file) # load each file 
#     trial_df = trial_to_df(file)  # convert to dataframe
#     print(trial_df.shape)
#     trials.append(trial_df)