# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet
"""
# Load libraries
import numpy as np
from numpy import random, newaxis
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv, DataFrame, concat, Series, merge
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
import os, time, itertools, random, logging
#from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, Conv1D, Activation, RepeatVector, TimeDistributed, Flatten, MaxPooling1D, ConvLSTM2D
#from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow_addons.metrics import F1Score
#from tensorflow.keras.utils import to_categorical
# import CategoricalAccuracy, CategoricalCrossentropy
#from tensorflow.compat.v1.keras.layers import mean_per_class_accuracy
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import keras.backend as K
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
from numba import cuda
#import pickle
import joblib
import seaborn as sns
import ray
from operator import itemgetter
import scipy
from scipy import stats
import threading
from threading import Thread
import concurrent.futures

# Data formatting
def split_dataset(data, years):
    """
    Training and test data split based on year. 
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
    lookback : lookback period
    n_output : prediction timesteps. The default is 7.
    window : sliding window size
    Returns
    -------
    X : features data in array format (input)
    y : target data in array format (output)
    dft: deterministic features for prediction timesteps in array formate

    """
    data = np.array(data) # convert data into numpy array
    X, y, dft = list(), list(), list() # get empty list for X (all features), y (targets), dft (deterministic features in prediction time steps)
    in_start = 0 # set start index as 0
	# step over the entire dataset one time step at a time
    for _ in range(len(data)):
        in_end = in_start + lookback # define the end of the input sequence
        out_end = in_end + n_output # define the end of the output sequence
		# ensure we have enough data left in the data and track for this instance 
        if out_end <= len(data) and len(set(TID[in_start:out_end])) == 1:
            X.append(data[in_start:in_end, :]) # append input sequence to features list
            y.append(data[in_end:out_end, 0:4]) # append output to the targets list
            dft.append(data[in_end:out_end,4:18]) # append the deterministic features for current timestep to the deterministic features list
        in_start += window # move along one time step
    X = np.array(X) # convert list to array
    y = np.array(y) # convert list to array
    dft = np.array(dft) # convert list to array
    a = np.where(np.min(y[:,:,0],axis = 1)==-1) # extract the indexes with rows that have unknown targets (i.e., values == -1)
    X = np.delete(X,a[0],axis =0) # delete unknown target rows
    y = np.delete(y,a[0],axis =0) # delete unknown target rows
    dft = np.delete(dft,a[0],axis =0) # delete unknown target rows
    if y.shape[1] == 1: # if the target is a single timestep
        y = y.reshape((y.shape[0],y.shape[2])) # then reshape the target 3D data to 2D
        dft = dft.reshape((dft.shape[0],dft.shape[2])) # also reshape the deterministic feat from 3D to 2D data 
    return X, y, dft 

def one_hot_decode(encoded_seq):
    """
    Reverse one_hot encoding
    Arguments:
        encoded_seq: array of one-hot encoded data 
	Returns:
		series of labels
	"""
    pred = [np.random.multinomial(1,vector) for vector in encoded_seq]
    return [np.argmax(vector) for vector in pred] # returns the index with the max value

def to_label(data):
    
    """
    Gets the index of the maximum value in each row. Can be used to transform one-hot encoded data to labels or probabilities to labels
    Parameters
    ----------
    data : one-hot encoded data or probability data

    Returns
    -------
    y_label : label encoded data

    """
    if len(data.shape) == 2: # if it is a one timestep prediction
        y_label = np.array(one_hot_decode(data)) # then one-hot decode to get the labels
    else: # otherwise 
        y_label = [] # create an empty list for the labels
        for i in range(data.shape[1]): # for each timestep
            y_lab = one_hot_decode(data[:,i,:]) # one-hot decode
            y_label.append(y_lab) # append the decoded value set to the list
        y_label = np.column_stack(y_label) # stack the sets in the list to make an array where each column contains the decoded labels for each timestep
    return y_label  # return the labels 

def algo_var(y_pred, y_label, test_dft, name, n_output = 1):
    """
    Assess model performance, generate metrics and output as a row of data. 
    Used to compare model performance between iterations

    Parameters
    ----------
    y_pred: predictions
    y_label: true values 
    test_dft: dataframe with deterministic features and identifiers
    name :  model name
    n_output : number of outputs
        DESCRIPTION. The default is 1.

    Returns
    -------
    datarow: output information for a single run 

    """
    cm = confusion_matrix(y_label, y_pred) # plot confusion matrix
    class_rep = classification_report(y_label,y_pred, zero_division = 0, output_dict = True) # generate classification reports
    class_rep = DataFrame(class_rep).transpose() # convert dictionary to dataframe

    # add y and ypred to the curent covariate features
    if n_output == 1:
        test_dft = np.column_stack((test_dft, y_label, y_pred))
    else:
        test_dft = np.append(test_dft, y_label, axis = 2)
        test_dft = np.append(test_dft, y_pred, axis = 2)
    
    datarow = [name,class_rep.iloc[4,0]] # add model name, accuracy (2)
    datarow.extend(class_rep.iloc[5,0:3].tolist()) # overall precision, recall and f1 score (3)
    f1 = class_rep.iloc[0:4,2].tolist()
    prec = class_rep.iloc[0:4,0].tolist()
    recall = class_rep.iloc[0:4,1].tolist()
    metrics_3 = [np.mean(itemgetter(0,1,3)(f1)), np.mean(itemgetter(0,1,3)(prec)),np.mean(itemgetter(0,1,3)(recall))]
    
    datarow.extend(f1) # add categorical f1 score (4)
    datarow.extend(prec) # add categorical precision (4)
    datarow.extend(recall) # add categorical recall (4)
    datarow.extend(metrics_3) # add metrics 3 (3)
    conmat = np.reshape(cm,(1,16)) # add confusion matrix values (16)
    datarow.extend(conmat.ravel().tolist())
    datarow.extend(np.sum(cm,0).tolist())
    t_prop = daily_dist(test_dft) # get the daily proportions
    for i in [0,1,2,3]: 
        ks = scipy.stats.ks_2samp(t_prop[t_prop['behavior'] == i].y_prop, t_prop[t_prop['behavior'] == i].ypred_prop, alternative = 'two_sided') # get the d statistics and p-values for the KS test
        datarow.extend(list(ks)) # add KS values (6)
    
    mean_df = t_prop.groupby('behavior').mean('y_prop')
    mean_val = mean_df.ypred_prop.values.tolist()
    datarow.extend(mean_val)
    print('working...')
    return datarow

def null_mod(train_y, y_label, test_dft, name):
    """
    Generate null0 model predictions, which are drawn from the overall behavioral frequency distributions
    Assess prediction performance

    Parameters
    ----------
    train_y : training y
    y_label : testing y in label format
    test_dft : deterministic features in testing dataset 
    name : model name

    Returns
    -------
    drow : performance metrics as a vector

    """
    train_ylab = to_label(train_y)
    train_prop = np.unique(train_ylab, return_counts = True)
    train_prop = train_prop[1]/len(train_ylab)
    y_pred = np.random.choice(a=range(4),size=len(y_label),replace = True, p=train_prop)
    drow = algo_var(y_pred, y_label, test_dft, name, 1)
    return drow

def transition_matrix(data):
    """
    Get transition matrix from data 
    Parameters
    ----------
    data : values 

    Returns
    -------
    M : transition matrixes 

    """
    transitions = data[:,1]
    predictor = data[:,0]
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,predictor):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def markov_null(train_y, train_X, y_label, test_X, test_dft, name):
    """
    Generate null1 model predictions, which are drawn from the transition likelihood between behaviors
    Assess prediction performance

    Parameters
    ----------
    train_y : training targets
    train_X : training features
    y_label : testing targets in label format
    test_dft : deterministic features in testing dataset 
    name : model name

    Returns
    -------
    drow : performance metrics as a vector

    """
    last_index = train_X.shape[1]-1
    
    a = np.where(train_X[:,last_index,0]==-1) # extract the indexes with rows that have unknown targets (i.e., values == -1)
    train_X = np.delete(train_X,a[0],axis =0) # delete unknown target rows
    train_y = np.delete(train_y,a[0],axis =0) # delete unknown target rows
    
    train_1 = to_label(train_y)
    train_0 = to_label(train_X[:,last_index,0:4])
    train_data = np.column_stack((train_0,train_1))
    
    test_1 = y_label
    test_0 = to_label(test_X[:,last_index,0:4])
    test_data = np.column_stack((test_0,test_1))
    test_data = np.column_stack((test_data, np.repeat(5, len(test_data))))
    
    train_mat = transition_matrix(train_data)
    
    for j in range(0,4):
        index = np.where(test_data[:,0] == j)
        index = index[0]
        test_data[index,2] = np.random.choice(range(4), len(index), replace = True, p = train_mat[j]) 
    
    y_pred = test_data[:,2]

    drow = algo_var(y_pred, y_label, test_dft, name, 1)
    return drow

# import datafile
dataset =  read_csv('data.csv', header = 0, index_col = 0) # VV only

n_input = 5
n_output = 1

train, test = split_dataset(dataset, 2015) # split data 
train_X, train_y, train_dft = to_supervised(train, train['TID'],1, n_input, n_output) # format training data
test_X, test_y, test_dft = to_supervised(test, test['TID'],n_output, n_input, n_output) # format testing data
y_label = to_label(test_y) # extract the labels for the test datar

# generate column names for metrics dataframe
colnames = ['model','accuracy','precision','recall','f1_score','f1_f','f1_r','f1_s','f1_t','precision_f',
            'precision_r','precision_s','precision_t','recall_f','recall_r','recall_s','recall_t','f1_3', 
            'precision_3','recall_3', 'FF','FR','FS','FT','RF','RR','RS','RT','SF','SR','SS','ST','TF',
            'TR','TS','TT','F_pred','R_pred','S_pred','T_pred','KSD_F','KSP_F','KSD_R','KSP_R','KSD_S',
            'KSP_S','KSD_T','KSP_T','F_prop','R_prop','S_prop','T_prop']

# create empty dataframes
null0_df = pd.DataFrame(columns = colnames)
null1_df = pd.DataFrame(columns = colnames)

# run analysis for  null0 model using threading
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:   
    results = [executor.submit(null_mod, train_y, y_label, test_dft, 'null') for i in range(1000)]
    for f in concurrent.futures.as_completed(results):
        null0_df.loc[len(null0_df.index)]= f.result()
finish = time.perf_counter()

null0_df.to_csv('null0_model_results.csv') # save results 

# run analyse for null1 model using threading
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:   
    results = [executor.submit(markov_null, train_y, train_X, y_label, test_X,test_dft,'null1') for i in range(1000)]
    for f in concurrent.futures.as_completed(results):
        null1_df.loc[len(null1_df.index)]= f.result()
finish = time.perf_counter()

null1_df.to_csv('null1_model_results.csv') # save results