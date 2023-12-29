# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:23:29 2021

@author: Jannet
"""

import numpy as np
from numpy import random, newaxis
from matplotlib import pyplot
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

#########################
#### Data formatting ####
#########################
# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

# import training datafiles
datasub =  read_csv('kian_trainset_focal.csv', header = 0, index_col = 0) # kian
datasub = datasub.drop(columns=['since_social','flower_shannon','fruit_shannon'], axis = 1) # get rid of these covariates
datasub = datasub[list(datasub.columns[0:18]) + list(datasub.columns[26:33]) + list(datasub.columns[18:26])] # reorder the covariates

def split_dataset(data, years):
    """
    Withing training, further split into training and testing data based on year
    Parameters
    ----------
    data : full dataset
    years : training-testing year cut off (training inclusive)
    -------
    train : Training data subset
    test : Testing data subset
    """
    train = data[data["year"] <= years] # get years that are equal or less than year used for training
    test = data[data["year"] > years] # get years that greater than year used for training cut off
    return train, test

def to_supervised(data, TID, window, lookback, n_output=7):
    """
    Format training data for multivariate, multistep time series prediction models

    Parameters
    ----------
    data : data
    TID : track identifier
    window : sliding window size
    lookback : lookback period
    n_output : prediction timesteps. The default is 7.
    Returns
    -------
    X : features data in array format (input)
    y : target data in array format (output)
    dft: deterministic features for prediction timesteps in array formate

    """
    # convert data into numpy array
    data = np.array(data) 
    # get empty list for X (all features), y (targets), dft (deterministic features in prediction time steps)
    # will be used to populate values 
    X, y, dft = list(), list(), list()
    # set start index as 0
    in_start = 0 
	# step over the entire dataset one time step at a time
    for _ in range(len(data)):
        in_end = in_start + lookback # define the end of the input sequence
        out_end = in_end + n_output # define the end of the output sequence
		# if we have enough data to create a full sequence and the track ID is the same for the entire sequence 
        if out_end <= len(data) and len(set(TID[in_start:out_end])) == 1:
            X.append(data[in_start:in_end, :]) # append input sequence to features list
            y.append(data[in_end:out_end, 0:4]) # append output to the targets list
            dft.append(data[in_end:out_end,4:18]) # append the deterministic features for current timestep to the deterministic features list
        in_start += window # move along the window time step
    # convert lists to array
    X = np.array(X) 
    y = np.array(y)
    dft = np.array(dft)
    # delete unknown target rows since we won't be predicting those
    a = np.where(np.min(y[:,:,0],axis = 1)==-1) # extract the indices with rows that have unknown targets (i.e., values == -1)
    X = np.delete(X,a[0],axis =0) # delete unknown target rows based on indices
    y = np.delete(y,a[0],axis =0) # delete unknown target rows
    dft = np.delete(dft,a[0],axis =0) # delete unknown target rows
    if y.shape[1] == 1: # if the target is a single timestep (i.e., predicting a single timestep at a time)
        y = y.reshape((y.shape[0],y.shape[2])) # then reshape the target 3D data to 2D
        dft = dft.reshape((dft.shape[0],dft.shape[2])) # also reshape the deterministic feat from 3D to 2D data 
    return X, y, dft 

def one_hot_decode(encoded_seq):
    """
    Reverse one_hot encoding
    Arguments:
        encoded_seq: array of one-hot encoded data 
	Returns:
		Series framed for supervised learning.
	"""
    if len(encoded_seq.shape) == 1: # if single prediction to decode
        return(np.argmax(encoded_seq))
    else: # otherwise there are multiple predictions to decode
        return [np.argmax(vector) for vector in encoded_seq] # returns the index with the max value

# from vanilla - used to subset from probs
# def one_hot_decode(encoded_seq):
#     """
#     Reverse one_hot encoding
#     Arguments:
#         encoded_seq: array of one-hot encoded data 
#  	Returns:
# 		series of labels
#  	"""
#     return(np.random.multinomial(1, encoded_seq, tuple))
   # return [np.argmax(vector) for vector in encoded_seq] # returns the index with the max value

# from ende - used to subset from probs
# def one_hot_decode(encoded_seq):
#     """
#     Reverse one_hot encoding
#     Arguments:
#         encoded_seq: array of one-hot encoded data 
# 	Returns:
# 		series of labels
# 	"""
#     pred = [np.random.multinomial(1,vector) for vector in encoded_seq]
#     return [np.argmax(vector) for vector in pred] # returns the index with the max value

########################
#### Model building ####
########################
def to_label(data, prob = False):
    
    """
    Gets the index of the maximum value in each row. Can be used to transform one-hot encoded data or probabilities to labels
    Parameters
    ----------
    data : one-hot encoded data or probability data
    prob : Boolean, False = get max value as label, True = sample from prob to get label

    Returns
    -------
    y_label : label encoded data

    """
    y_label = [] # create empty list for y_labels
    if len(data.shape) == 2: # if it is a one timestep prediction 
        if prob == False: # and prob is false
            y_label = np.array(one_hot_decode(data)) # then one-hot decode to get the labels
        else:
            for i in range(data.shape[0]):
                y_lab = np.random.multinomial(1, data[i,:])
                y_label.append(y_lab) # append the decoded value set to the list
            y_label = np.array(y_label)
            y_label = one_hot_decode(y_label)
    else: # otherwise 
        if prob == False:    
            for i in range(data.shape[1]): # for each timestep
                y_lab = one_hot_decode(data[:,i,:]) # one-hot decode
                y_label.append(y_lab) # append the decoded value set to the list
                y_label = np.column_stack(y_label) # stack the sets in the list to make an array where each column contains the decoded labels for each timestep
        else:
            for i in range(data.shape[1]): # for each timestep
                y_lab = np.random.multinomial(1, data[:,i,:])
                y_lab = one_hot_decode(y_lab) # one-hot decode
                y_label.append(y_lab) # append the decoded value set to the list
                y_label = np.column_stack(y_label) # stack the sets in the list to make an array where each column contains the decoded labels for each timestep
    return y_label  # return the labels 

def get_sample_weights(train_y, weights):
    """
    Get sample weights for the for training data, since tensorflow built-in 3D class weights is not supported 

    Parameters
    ----------
    train_y : training targets 
    weights : weight dictionary 
    Returns
    -------
    train_lab : array of sample weights for each label at each timestep

    """
    train_lab = to_label(train_y) # get the one-hot decoded labels for the training targets
    train_lab = train_lab.astype('float64') # convert the datatype to match the weights datatype
    train_labels = np.copy(train_lab)
    for key, value in weights.items(): # replace each label with its pertaining weight according to the weights dictionary
        train_labels[train_lab == key] = value
    train_labels[train_labels == 0] = 0.0000000000001 # weight cannot be 0 so reassign to very small value if it is
    return train_labels # return the formatted sample weights
    
def f1(y_true, y_pred):
    '''
    calculate f1 metric within tensorflow keras framework    

    Parameters
    ----------
    y_true : observed value 
    y_pred : predicted value 

    Returns
    -------
    f1-score
    '''
    y_pred = K.round(y_pred) # round to get prediction from probability 
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0) # calculate true positive
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0) # calculate false positive
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) # calculate false negative
    p = tp / (tp + fp + K.epsilon()) # calculate precision
    r = tp / (tp + fn + K.epsilon()) # calculate recall
    f1 = 2*p*r / (p+r+K.epsilon()) # calculate f1-score
    f1 = tf.where(tf.math.is_nan(f1), # if nan
                  tf.zeros_like(f1), # set to 0
                  f1) # else set to f1 score
    return K.mean(f1) # return the mean f1-score

def f1_loss(y_true, y_pred):
    '''
    calculate loss f1 metric within tensorflow keras framework

    Parameters
    ----------
    y_true : observed value 
    y_pred : predicted value 

    Returns
    -------
    loss f1 metric
    '''
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0) # calculate true positive
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0) # calculate false positive
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) # calculate false negative
    p = tp / (tp + fp + K.epsilon()) # calculate precision
    r = tp / (tp + fn + K.epsilon()) # calculate recall
    f1 = 2*p*r / (p+r+K.epsilon()) # calculate f1-score
    f1 = tf.where(tf.math.is_nan(f1), # is nan
                  tf.zeros_like(f1),# set to 0
                  f1) # else set to f1 score
    return 1 - K.mean(f1) # calculate loss f1

def eval_f1_iter(model, params, train_X, train_y, test_X, test_y, patience=50, max_epochs = 300, atype = 'VRNN', n = 1):
    """
    Fit and evaluate model n number of times. Get the average of those runs

    Parameters
    ----------
    model : model
    params : hyperparameters
    train_X : training features
    train_y :  training targets
    test_X : testing features
    test_y : testing targets
    patience: early stopping patience value
    max_epochs: number of epochs to run
    atype: architecture type (VRNN or ENDE)
    n : number of iterations to run for a single model

    Returns
    -------
    eval_run : metrics for each iteration
    avg_val: average of the metrics average: val_f1, val_loss, train_f1, train_loss 
    """
    # assign class weights 
    weights = dict(zip([0,1,2,3], 
                       [params['weights_0'], 
                        params['weights_1'], 
                        params['weights_2'], 
                        params['weights_3']]))
    # assign the callback and weight type based on the model type
    if atype == 'VRNN':
        class_weights = weights # assign class weights as weights
        sample_weights = None
    elif atype == 'ENDE':
        class_weights = None 
        total = sum(weights.values()) # get the sum of the weights to normalize
        sample_weights = {ky: val / total for ky, val in weights.items()} # get the sample weight values
        sample_weights = get_sample_weights(train_y, sample_weights) # generate the formatted sample weights 
    else:
        raise Exception ('invalid model type')
    early_stopping = EarlyStopping(patience= patience, monitor='val_loss', mode = 'min', restore_best_weights=True, verbose=0)
    eval_run = [] # create empty list for the evaluations
    for i in range(n): # for each iteration
        # fit the model 
        history = model.fit(train_X, 
                            train_y, 
                            epochs = max_epochs, 
                            batch_size = params['batch_size'],
                            verbose = 2,
                            shuffle=False,
                            validation_data = (test_X, test_y),
                            sample_weight = sample_weights,
                            class_weight = class_weights,
                            callbacks = [early_stopping])
        # pull out monitoring metrics
        if len(history.history['loss']) == max_epochs: # if early stopping not activated then
            params['epochs'] = max_epochs # assign the epochs to the maximum epochs
            val_loss = history.history['val_loss'][-1] # get the last val loss
            train_loss = history.history['loss'][-1] # get the last train loss
            f1 = history.history['val_f1'][-1] # pull the last validation f1 from the history
            train_f1 = history.history['f1'][-1] # pull the last train f1 from the history
        else: # otherwise if early stopping was activate
            params['epochs'] = len(history.history['loss'])-patience # assign stopping epoch as the epoch before improvements dropped
            val_loss = history.history['val_loss'][-patience-1] # assign validation loss from the stopping epochs
            train_loss = history.history['loss'][-patience-1] # assign trainn loss from the stopping epochs
            f1 = history.history['val_f1'][-patience-1] # assign the validation f1 from the stopping epoch
            train_f1 = history.history['f1'][-patience-1] # assign the training f1 from the stopping epoch
        eval_run.append([f1,val_loss,train_f1,train_loss])
    avg_val = np.mean(eval_run,axis=0)
    return eval_run, avg_val[0], avg_val[1], avg_val[2], avg_val[3]

# def eval_f1_iter_ende(model, params, train_X, train_y, test_X, test_y, patience=50, max_epochs = 300, atype = 'VRNN', n = 1):
#     """
#     Fit and evaluate model n number of times. Get the average of those runs

#     Parameters
#     ----------
#     model : model
#     params : hyperparameters
#     train_X : training features
#     train_y :  training targets
#     test_X : testing features
#     test_y : testing targets
#     patience: early stopping patience value

#     max_epochs: number of epochs to run
#     atype: architecture type (VRNN or ENDE)
#     n : number of iterations
    
#     Returns
#     -------
#     eval_run : metrics for each iteration
#     avg_val: average of the metrics average: val_f1, val_loss, train_f1, train_loss 
#     """
#     # assign the weights 
#     weights = dict(zip([0,1,2,3], [params['weights_0'], params['weights_1'], params['weights_2'], params['weights_3']]))
#     early_stopping = EarlyStopping(patience= patience, monitor='val_loss', mode = 'min', restore_best_weights=True, verbose=0) 
#     # assign the callback and weight type based on the model type
#     if atype == 'VRNN':
#         early_stopping = EarlyStopping(patience= patience, monitor='val_f1_score', mode = 'max', restore_best_weights=True, verbose=0)
#         class_weights = weights # assign class weights as weights
#         sample_weights = None
#         calls = [early_stopping]
#     else:
#      #   early_stopping = F1EarlyStopping(validation_data=[test_X, test_y], train_data=[train_X, train_y], patience= patience)
#        # f1_metric = F1Metrics(validation_data=[test_X, test_y], train_data=[train_X, train_y])
#         class_weights = None 
#         total = sum(weights.values()) # get the sum of the weights to normalize
#         sample_weights = {ky: val / total for ky, val in weights.items()} # get the sample weight values
#         sample_weights = get_sample_weights(train_y, weights) # generate the formatted sample weights 
#         calls = [early_stopping, f1_metric]
#     eval_run = [] # create empty list for the evaluations
#     for i in range(n): # for each iteration
#         # fit the model 
#         history = model.fit(train_X, train_y, 
#                             epochs = max_epochs, 
#                             batch_size = params['batch_size'],
#                             verbose = 2,
#                             shuffle=False,
#                             validation_data = (test_X, test_y),
#                             sample_weight = sample_weights,
#                             class_weight = class_weights,
#                             callbacks = [early_stopping])
#         # pull out monitoring metrics
#         if len(history.history['loss']) == max_epochs:
#             params['epochs'] = max_epochs # assign the epochs to the maximum epochs
#             val_loss = history.history['val_loss'][-1] # get the last val loss
#             train_loss = history.history['loss'][-1] # get the last train loss
#             if atype == 'ENDE': # if the model is an encoder-decoder
#                 f1 = early_stopping.val_f1s[-1] # then pull the last validation f1 from the early stopping metric
#                 train_f1 = early_stopping.train_f1s[-1] # pull the last training f1 from the early stopping metric
#                 f1 = f1_metric['val_f1'][-1] # pull the last validation f1 from the history
#               #  train_f1 = history.history['get_f1'][-1] # pull the last train f1 from the history
#             else: # otherwise 
#                 f1 = history.history['val_f1_score'][-1] # pull the last validation f1 from the history
#                 train_f1 = history.history['f1_score'][-1] # pull the last train f1 from the history
#         else: # otherwise if early stopping was activate
#             params['epochs'] = len(history.history['loss'])-patience # assign stopping epoch as the epoch before no more improvements were seen in the f1 score
#             val_loss = history.history['val_loss'][-patience-1] # assign validation loss from the stopping epochs
#             train_loss = history.history['loss'][-patience-1] # assign trainn loss from the stopping epochs
#             if atype == 'ENDE': # if the model is the encoder-decoder
#                 f1 = history.history['val_get_f1'][-patience-1] # assign the training f1 from the stopping epoch
#                 train_f1 = history.history['get_f1'][-patience-1] # assign the training f1 from the stopping epoch
#             else: # otherwise
#                 f1 = history.history['val_f1_score'][-patience-1] # assign the training f1 from the stopping epoch
#                 train_f1 = history.history['f1_score'][-patience-1] # assign the training f1 from the stopping epoch
#         eval_run.append([np.nan,val_loss,np.nan,train_loss])
#    # avg_val = np.mean(eval_run,axis=0)
#     return eval_run, np.nan, val_loss, np.nan, train_loss

def build_rnn(train_X, train_y, neurons_n=10, hidden_n=10, lr_rate=0.001, d_rate = 0.2, layers = 1, mtype = 'LSTM'):
    """
    Vanilla LSTM for single timestep output prediction (one-to-one or many-to-one)
    Parameters
    ----------
    neurons_n : int,number of neurons, (the default is 10).
    hidden_n : int, number of hidden neurons, (the default is 10).
    lr_rate : float, learning rate (the default is 0.001).
    d_rate : float, drop out rate (the default is 0.2).
    layers: 1 
    mtype: string, model type (LSTM or GRU only, default is LSTM)
    Returns
    -------
    model : model
    """
    features = train_X.shape[2] # get the number of features
    lookback = train_X.shape[1] # get the lookback period
    targets = train_y.shape[1] # get the target number
    
    model = Sequential() # create an empty sequential shell 
    model.add(Masking(mask_value = -1, input_shape = (lookback, features), name = 'Masking')) # add a masking layer to tell the model to ignore missing values (i.e., values of -1)
    if mtype == 'LSTM': # if the model type is LSTM
        model.add(LSTM(units =neurons_n, input_shape = (lookback,features), name = 'LSTM')) # set the RNN type
    else:
        model.add(GRU(units =neurons_n, input_shape = (lookback,features), name = 'GRU')) # set the RNN type
    for i in range(layers): # for each hidden layer 
        model.add(Dense(units = hidden_n, activation = 'relu', kernel_initializer =  'he_uniform')) # add dense layer
        model.add(Dropout(rate= d_rate)) # add dropout
    model.add(Dense(units = targets, activation = "softmax", name = 'Output')) # add output layer
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = lr_rate), metrics= [F1Score(num_classes=targets, average = 'macro'),'accuracy']) # compile model, set learning rate and metrics
    return model 

def hyp_rnn_nest(params, features, targets):
    """
    Generate vanilla RNN for single timestep output prediction (one-to-one or many-to-one)
    With nested architecture
    
    Parameters
    ----------
    params: hyperparameter search space
    features: number of features
    targets: number of targets
    
    Returns
    -------
    model : model
    """ 
    lookback = params['lookback'] # assign lookback
    model = Sequential() # create an empty sequential shell 
    # add a masking layer to ignore missing values
    model.add(Masking(mask_value = -1, 
                      input_shape = (lookback, features), 
                      name = 'Masking')) 
    # set the RNN type
    if params['mtype']=='LSTM': # if the model is an LSTM
        model.add(LSTM(units =params['neurons_n'], 
                       input_shape = (lookback,features), 
                       return_sequences= False, 
                       name = 'LSTM')) 
    elif params['mtype']=='GRU': # else if the model is a GRU, set architecture accordingly
        model.add(GRU(units =params['neurons_n'], 
                      input_shape = (lookback,features), 
                      return_sequences= False,
                      name = 'GRU'))
    else: # otherwise
        raise Exception ('invalid model architecture')
    model.add(Dropout(rate= params['drate'])) # add drop out
    for i in range(params['hidden_layers']):# increase complexity by adding hidden layers
        # add dense layer
        model.add(Dense(units = params['hidden_n'+str(i)], 
                        activation = 'relu', 
                        kernel_initializer = 'he_uniform')) 
        model.add(Dropout(rate= params['drate'])) # add another dropout layer
    # add output layer
    model.add(Dense(units = targets, 
                    activation = "softmax", 
                    name = 'Output')) 
    # compile the model based on loss function
    if params['loss'] == 'f1_loss':    
        model.compile(loss = f1_loss, 
                      optimizer = Adam(learning_rate = params['learning_rate']), 
                      sample_weight_mode = 'temporal',
                      metrics=[f1]) 
        # model.compile(loss = f1_loss, optimizer = Adam(learning_rate = params['learning_rate']), sample_weight_mode = 'temporal',metrics=[f1,'Accuracy', 'Precision', 'Recall', AUC(curve = 'ROC', name = 'ROC'), AUC(curve='PR',name ='PR')]) # compile the model
    else:
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = Adam(learning_rate = params['learning_rate']), 
                      metrics = [F1Score(num_classes=4, average = 'macro')]) # compile the model
        #  model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = params['learning_rate']), metrics = [F1Score(num_classes=4, average = 'macro'), 'Accuracy', 'Precision', 'Recall', AUC(curve = 'ROC', name = 'ROC'), AUC(curve='PR',name ='PR')]) # compile the model
    return model 

###########################
#### Hyperoptimization ####
###########################
def run_trials(filename, objective, space, rstate, initial = 20, trials_step = 1):
    """
    Run and save trials indefinitely until manually stopped. 
    Used to run trials in small batches and periodically save to file.
    
    Parameters
    ----------
    filename : trial filename
    objective : objective
    space : parameters
    rstate: set random state for consistency across trials
    initial: initial number of trials, should be > 20 
    trials_steps: how many additional trials to do after loading saved trials.
    
    Returns
    -------
    None.

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

def hyperoptimizer_vrnn(params):
    """
    hyperparameter optimizer objective function to be used with hyperopt

    Parameters
    ----------
    params : hyperparameter search space

    Returns
    -------
    dict
        loss: loss value to optimize through minimization (i.e., validation loss)
        status: default value for hyperopt
        params: the hyperparameter values being tested
        val_loss: validation loss
        train_loss: training loss
        train_f1: training f1

    """
    targets=4 # set number of targets (4 behavior classes)
    train, test = split_dataset(datasub, 2015) # split the data by year
    # format training data
    train_X, train_y, train_dft = to_supervised(data = train.iloc[:,7:33], 
                                                TID = train['TID'], 
                                                window = 1, 
                                                lookback = params['lookback'], 
                                                n_output=params['n_output']) 
    # format testing data
    test_X, test_y, test_dft = to_supervised(data = test.iloc[:,7:33], 
                                             TID = test['TID'],
                                             window = 1, 
                                             lookback = params['lookback'], 
                                             n_output = params['n_output'])
    # assign and format feature set
    if params['predictor'] == 'full': # use full set of features
        features=26 # set feature number
    elif params['predictor'] == 'behavior': # use only prior behaviors as features
        features=4 # set features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:4]) 
        test_X = np.copy(test_X[:,:,0:4])
    else: # use the extrinsic conditions
        features = 17
        # subset only extrinsic features
        train_X = np.copy(train_X[:,:,np.r_[4:9,11:18,21:26]])
        test_X = np.copy(test_X[:,:,np.r_[4:9,11:18,21:26]])    
    model = hyp_rnn_nest(params, features, targets) # build model
    # fit model and extract monitoring metrics
    _, val_f1,val_loss,train_f1, train_loss = eval_f1_iter(model, 
                                                           params, 
                                                           train_X, 
                                                           train_y, 
                                                           test_X, 
                                                           test_y, 
                                                           patience = 30, 
                                                           atype ='VRNN', 
                                                           max_epochs = 200, 
                                                           n=1) 
    print('Best validation for trial:', val_f1) # print the validation score
    return {'loss': -val_f1,
            'status': STATUS_OK,  
            'params': params,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'train_loss': train_loss,
            'train_f1':train_f1}   

space_vrnn = {'covariate'              : 'full',
              'drate'                  : hp.quniform('drate',0.1,0.9,0.1),
              'neurons_n'              : scope.int(hp.quniform('neurons_n',5,50,5)),
              'n_output'               : 1,
              'learning_rate'          : 0.001,
              'hidden_layers'          : scope.int(hp.choice('layers',[0,1])),
              'hidden_n0'              : scope.int(hp.quniform('hidden_n0',5,50,5)),
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'epochs'                 : 200,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,5,0.5),
              'weights_1'              : 1,
              'weights_2'              : scope.int(hp.quniform('weights_2',1,25,1)),
              'weights_3'              : scope.int(hp.quniform('weights_3',1,10,1)),
              'mtype'                  : 'LSTM'
              }

space_vrnn = {'covariate'              : 'full',
              'drate'                  : hp.quniform('drate',0.1,0.5,0.1),
              'neurons_n'              : scope.int(hp.quniform('neurons_n',5,50,5)),
              'n_output'               : 1,
              'learning_rate'          : 0.001,
              'hidden_layers'          : 0,
              'hidden_n0'              : 0,
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'epochs'                 : 200,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,3,0.5),
              'weights_1'              : 1,
              'weights_2'              : hp.quniform('weights_2',1,12,1),
              'weights_3'              : hp.quniform('weights_3',1,5,0.5),
              'mtype'                  : 'GRU'
              }

# ENDE
def hyp_ende_nest(params, features, targets):
    """
    Encoder-decoder LSTM for single timestep output prediction (one-to-one or many-to-one)
    With nested architecture
    Parameters
    ----------
    params: hyperparameters 
    features: number of features
    targets: number of targets
    Returns
    -------
    model : model
    """ 
    lookback = params['lookback'] # extract lookbackf
    n_outputs = params['n_output'] # extract output
    model = Sequential() # create an empty sequential shell 
    model.add(Masking(mask_value = -1, input_shape = (lookback, features), name = 'Masking')) # add a masking layer to tell the model to ignore missing values
    if params['mtype']=='LSTM': # if the model is an LSTM
        model.add(LSTM(units =params['neurons_n0'], input_shape = (lookback,features), return_sequences= False, name = 'LSTM')) # set the RNN type
        model.add(Dropout(rate= params['drate'])) # add another drop out
        for i in range(params['hidden_layers']): # increase the model complexity through adding more hidden layers 
             model.add(Dense(units = params['hidden_n'+str(i)], activation = 'relu', kernel_initializer =  'he_uniform')) # add dense layer
             model.add(Dropout(rate= params['drate'])) # add dropout rate
    else: # the model is a GRU, set architecture accordingly
        model.add(GRU(units =params['neurons_n0'], input_shape = (lookback,features), return_sequences= False)) # set the RNN type
        model.add(Dropout(rate= params['drate'])) # add another drop out
        for i in range(params['hidden_layers']):# increase complexity through hidden layers
            model.add(Dense(units = params['hidden_n'+str(i)], activation = 'relu', kernel_initializer = 'he_uniform')) # add dense layer
            model.add(Dropout(rate= params['drate'])) # add dropout layer
    model.add(RepeatVector(n_outputs)) # repeats encoder context for each prediction timestep
    if params['mtype']== 'LSTM': # if the model type is LSTM 
        model.add(LSTM(units= params['neurons_n1'], input_shape = (lookback,features), return_sequences=True)) # set the RNN type
    else: # else set the layer to GRU
        model.add(GRU(units =params['neurons_n1'], input_shape = (lookback,features), return_sequences = True)) # set the RNN type
    model.add(TimeDistributed(Dense(units = params['td_neurons'], activation='relu'))) # used to make sequential predictions, applies decoder fully connected layer to each prediction timestep
    model.add(TimeDistributed(Dense(targets, activation = "softmax"))) # applies output layer to each prediction timestep
    model.compile(loss = f1_loss, optimizer = Adam(learning_rate = params['learning_rate']), sample_weight_mode = 'temporal',metrics=[f1]) # compile the model
  #  model.compile(loss = f1_loss, optimizer = Adam(learning_rate = params['learning_rate']), sample_weight_mode = 'temporal',metrics=[f1,'Accuracy','Precision', 'Recall', AUC(curve = 'ROC', name = 'ROC'), AUC(curve='PR',name ='PR')]) # compile the model
    #model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = params['learning_rate']), sample_weight_mode = 'temporal', metrics = f1_score) # compile the model
   # model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = params['learning_rate']), sample_weight_mode = 'temporal', metrics = ['CategoricalAccuracy', 'Accuracy', 'Precision', 'Recall', AUC(curve = 'ROC', name = 'ROC'), AUC(curve='PR',name ='PR')]) # compile the model
    return model 

def hyperoptimizer_ende(params):
    """
    hyperparameter optimizer objective function to be used with hyperopt

    Parameters
    ----------
    params : hyperparameter search space
    
    Returns
    -------
    dict
        loss: loss value to optimize through minimization (i.e., -validation f1)
        status: default value for hyperopt
        params: the hyperparameter values being tested
        val_loss: validation loss
        train_loss: training loss
        train_f1: training f1

    """
    targets=4 # set targets
    train, test = split_dataset(datasub, 2015) # split the data
    train_X, train_y, train_dft = to_supervised(data = train.iloc[:,7:33], TID = train['TID'], window = 1, lookback = params['lookback'], n_output=params['n_output']) # format training data
    test_X, test_y, test_dft = to_supervised(data = test.iloc[:,7:33], TID = test['TID'],window = params['n_output'], lookback = params['lookback'], n_output = params['n_output']) # format testing data
    # format target as 3D array if it isn't already
    if params['n_output'] == 1:
        test_y = test_y[:,newaxis,:]
        train_y = train_y[:,newaxis,:]
    
    # extract appropriate covariates
    if params['covariate'] == 'full':
        features=26 # set features
    elif params['covariate'] == 'behavior':    
        features=4 # set features
        train_X = np.copy(train_X[:,:,0:4])
        test_X = np.copy(test_X[:,:,0:4])
    else:
        features = 17
        train_X = np.copy(train_X[:,:,np.r_[4:9,11:18,21:26]])
        test_X = np.copy(test_X[:,:,np.r_[4:9,11:18,21:26]])   
       
    model = hyp_ende_nest(params, features, targets) # build model based on hyperparameters
    
    _, val_f1,val_loss,train_f1, train_loss = eval_f1_iter(model, params, train_X, train_y, test_X, test_y, patience = 30, atype ='ENDE', max_epochs = 200, n=1) # fit model and extract monitoring metrics
    print('Best validation for trial:', val_f1) # print the validation score
    return {'loss': val_loss,
            'status': STATUS_OK,  
            'params': params,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'train_loss': train_loss,
            'train_f1':train_f1}    

##########################
#### Model evaluation ####
##########################
def monitoring_plots(result):
    """
    plot the training and validation loss, f1 and accuracy

    Parameters
    ----------
    result : history from the fitted model 

    Returns
    -------
    monitoring plots outputted
    """
    fig, ax = pyplot.subplots(1,3, figsize = (8,2))
    # plot the loss
    pyplot.subplot(1,3,1)
    pyplot.plot(result.history['loss'], label='train')
    pyplot.plot(result.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.title('loss')
        
    # plot the f1 score
    pyplot.subplot(1,3,2)
    pyplot.plot(result.history['f1_score'], label='train')
    pyplot.plot(result.history['val_f1_score'], label='validation')
    pyplot.legend()
    pyplot.title('f1')
    
    # plot the accuracy score
    pyplot.subplot(1,3,3)
    pyplot.plot(result.history['accuracy'], label='train')
    pyplot.plot(result.history['val_accuracy'], label='validation')
    pyplot.legend()
    pyplot.title('accuracy')
    fig.tight_layout()
    return fig
    
def confusion_mat(y, y_pred):
    """
    generates and visualizes the confusion matrix

    Parameters
    ----------
    y : labeled true values
    y_pred : labeled predictions

    Returns
    -------
    cm : confusion matrix

    """
    labels = ['feeding','resting','socializing','traveling']
    fig, ax = pyplot.subplots(1,2,figsize=(6,2))
    cm_norm = confusion_matrix(y,y_pred, normalize = 'true') # normalized confusion matrix to get proportion instead of counts
    cm_count = confusion_matrix(y,y_pred) # get confusion matrix without normalization (i.e., counts)
    sns.set(font_scale=0.5)
    pyplot.subplot(1,2,1)
    sns.heatmap(cm_count, 
                    xticklabels=labels, 
                    yticklabels=labels, 
                    annot=True, 
                    fmt ='d') 
    pyplot.yticks(rotation=0) 
    pyplot.xticks(rotation=0) 
    pyplot.ylabel('True label', fontsize = 7)
    pyplot.xlabel('Predicted label', fontsize = 7)
    pyplot.subplot(1,2,2)
    sns.heatmap(cm_norm, 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=True) 
    pyplot.yticks(rotation=0) 
    pyplot.xticks(rotation=0) 
    pyplot.ylabel('True label', fontsize = 7)
    pyplot.xlabel('Predicted label', fontsize = 7)
    fig.tight_layout()
    return fig

def class_report(y, y_pred):
    """
    generate the class report to get the recall, precision, f1 and accuracy per class and overall

    Parameters
    ----------
    y : labeled true values
    y_pred : labeled predictions

    Returns
    -------
    class_rep : classification report

    """
    class_rep = classification_report(y,y_pred, zero_division = 0, output_dict = True) # generatate classification report as dictionary
    class_rep = DataFrame(class_rep).transpose() # convert dictionary to dataframe
    return class_rep    

def result_summary(test_y, y_prob, path, filename):
    """
    Summary of model evaluation for single model for multiple iterations. Generates the prediction timestep and overall F1 score, overall 
    classification report and confusion matrix. Outputs classification report nad confusion matrix to pdf.
    
    Parameters
    ----------
    test_y: one-hot encoded test_y
    y_prob: probability of class prediction 

    Returns
    -------
    score: overall f1 score
    scores: timestep level f1 scores
    class_rep: overall classification report
    cm: overall confusion matrix

    """
    y_label = to_label(test_y) # observed target
    y_pred = to_label(y_prob, prob = True) # predicted target
    
    if len(y_label.shape) == 1: # no multiple scores
        scores = 'nan'
    else:
        scores = [] # create empty list to populate with timestep level predictions
        for i in range(y_pred.shape[1]): # for each timestep
            f1 = f1_score(y_label[:,i], y_pred[:,i], average = 'macro') # get the f1 value at the timestep
            scores.append(f1) # append to the empty scores list
        y_pred = np.concatenate(y_pred) # merge predictions across timesteps to single vector
        y_label = np.concatenate(y_label) # merge target values across timesteps to single vector
    print('sequence level f1 score: ', scores)
    score = f1_score(y_label, y_pred, average = 'macro') # generate the overall f1 score
    print('overall f1 score: ', score)
    
    class_rep = class_report(y_label, y_pred) # get class report for overall
    
    with PdfPages(path+filename+'.pdf') as pdf:
        cm = confusion_mat(y_label, y_pred) # get confusion matrix for overall
        pdf.savefig(cm) # save figure
        pyplot.close() # close page
        pyplot.figure(figsize=(6, 2)) # assign figure size
        pyplot.table(cellText=np.round(class_rep.values,4),
                      colLabels = class_rep.columns, 
                      rowLabels=class_rep.index,
                      loc='center',
                      fontsize = 9)
        pyplot.axis('tight') 
        pyplot.axis('off')
        pdf.savefig() # save figure
        pyplot.close() # close page
    return score, scores, class_rep, cm