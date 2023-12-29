# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

Run single models of encoder-decoder RNN using GRU and LSTM architecture and evaluate the results. This code is implemented within the training phase of the project.
This code can be used to manually tune parameters by monitoring the loss plots, confusion matrices and classification reports. 

@author: Jannet
"""
# Load libraries
import os
import time
import numpy as np
from numpy import newaxis
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv, DataFrame
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, RepeatVector, TimeDistributed
#from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.callbacks import Callback

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
# get rid of these predictors
datasub = datasub.drop(columns=['since_social','flower_shannon','fruit_shannon'], 
                       axis = 1) 
# reorder the predictors
datasub = datasub[list(datasub.columns[0:18]) + list(datasub.columns[26:33]) + list(datasub.columns[18:26])]

#####################
#### Data Format ####
#####################
# assign input and output 
n_input = 5
n_output = 1

# split dataset
train, test = bmf.split_dataset(datasub, 2015)

# format the training and testing data for the model
train_X, train_y, train_dft = bmf.to_supervised(data = train.iloc[:,7:33], 
                                                TID = train['TID'], 
                                                window = 1, 
                                                lookback = n_input, 
                                                n_output= n_output)
     
test_X, test_y, test_dft = bmf.to_supervised(data = test.iloc[:,7:33], 
                                             TID = test['TID'], 
                                             window = 1, 
                                             lookback = n_input, 
                                             n_output= n_output)



def report_average(reports):
    """
    get report average for classification report across multiple runs 

    Parameters
    ----------
    reports : list of classification reports 

    Returns
    -------
    mean_dict : dictionary of the average classification values

    """
    mean_dict = dict() # create an empty dictionary
    for label in reports[0].keys(): # for each key 
        dictionary = dict() # create a dictionary
        if label in 'accuracy': # for the accuracy take the average across all reports
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue
        for key in reports[0][label].keys(): # for other keys
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports) # get the average of the values
        mean_dict[label] = dictionary # add dictionary to mean_dict
    return mean_dict

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
    # plot the loss
    pyplot.plot(result.history['loss'], label='train')
    pyplot.plot(result.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.title('loss')
    pyplot.show()
        
    try:
        # plot the f1 score
        pyplot.plot(result.history['f1_score'], label='train')
        pyplot.plot(result.history['val_f1_score'], label='validation')
        pyplot.legend()
        pyplot.title('f1')
        pyplot.show()
       
    except:
        pyplot.plot(early_stopping.train_f1s, label='train')
        pyplot.plot(early_stopping.val_f1s, label='validation')
        pyplot.legend()
        pyplot.title('f1')
        pyplot.show()
     
def result_summary(test_y, y_prob):
    """
    Summary of model evaluation for single model for multiple iterations. Generates the prediction timestep and overall F1 score, overall 
    classification report and confusion matrix
    
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
    y_label = to_label(test_y)
    y_pred = to_label(y_prob)
    if len(y_label.shape) == 1:
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
    cm = confusion_mat(y_label, y_pred) # get confusion matrix for overall
    return score, scores, class_rep, cm

def build_ende(train_X, train_y, layers = 2, neurons_n = 50, hidden_n = 20, td_neurons = 20, lr_rate  = 0.001, d_rate = 0.8, mtype = 'LSTM'):
    """
    Single encoder-decoder model

    Parameters
    ----------
    train_X : training features
    train_y : training predictions
    layers : number of layers The default is 2.
    neurons_n : number of neurons. The default is 50.
    hidden_n : number of hidden neurons. The default is 20.
    td_neurons : number of timde distributed neurons. The default is 20.
    lr_rate : Learning rate. The default is 0.001.
    d_rate : dropout rate. The default is 0.8.
    mtype : model type, should be LSTM or GRU. The default is 'LSTM'.

    Returns
    -------
    model : compiled model

    """
    lookback = train_X.shape[1] # set lookback
    features = train_X.shape[2] # set features
    n_outputs = train_y.shape[1] # set prediction timesteps
    targets = train_y.shape[2] # set number of targets per timesteps
	# define model
    model = Sequential() # create empty sequential model
    model.add(Masking(mask_value = -1, input_shape = (lookback, features), name = 'Masking')) # add a masking layer to tell the model to ignore missing values
    if mtype == 'LSTM': # if the model is an LSTM
        model.add(LSTM(units =neurons_n, input_shape = (lookback,features), name = 'LSTM')) # set the RNN type
    else: # otherwise set the GRU as the model type
        model.add(GRU(units =neurons_n, input_shape = (lookback,features), name = 'GRU')) # set the RNN type
    for i in range(layers): # for each layer  
        model.add(Dense(units = hidden_n, activation = 'relu', kernel_initializer =  'he_uniform')) # add a dense layer 
        model.add(Dropout(rate= d_rate)) # and add a dropout layer
    model.add(RepeatVector(n_outputs)) # repeats encoder context for each prediction timestep
    if mtype == 'LSTM': # if the model type is LSTM 
        model.add(LSTM(units =neurons_n, input_shape = (lookback,features), return_sequences=True)) # set the RNN type
    else: # else set the layer to GRU
        model.add(GRU(units =neurons_n, input_shape = (lookback,features), return_sequences = True)) # set the RNN type
    model.add(TimeDistributed(Dense(units = td_neurons, activation='relu'))) # used to make sequential predictions, applies decoder fully connected layer to each prediction timestep
    model.add(TimeDistributed(Dense(targets, activation = "softmax"))) # applies output layer to each prediction timestep
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = lr_rate), metrics = 'accuracy', sample_weight_mode = 'temporal') # compile the model
    return model

# generate sample weights as inverse proportion of frequency in dataset
weights = dict(zip([0,1,2,3], [34,11,22,45])) # create a dictionary with the weights 
sample_weights = get_sample_weights(train_y, weights) # get sample weights

mod_ende = build_ende(train_X, train_y, layers = 2, neurons_n = 50, hidden_n = 20, td_neurons = 20, lr_rate  = 0.001, d_rate = 0.8, mtype = 'LSTM')
early_stopping = F1EarlyStopping(validation_data = [test_X, test_y], train_data = [train_X, train_y], patience=5)
start_time = time.time() # generate the start time to keep track of run time
results = mod_ende.fit(train_X, train_y, 
                       validation_data = (test_X,test_y),
                       epochs = 4, 
                       batch_size = 128,
                       shuffle=False,
                       sample_weight = sample_weights,
                       class_weight = None,
                       verbose = 2,
                       callbacks = [early_stopping])
print('took', (time.time()-start_time)/60, ' minutes') # print the time lapsed 

monitoring_plots(results)
y_prob = mod_ende.predict(test_X) # get predictions
score, scores, class_rep, cm = result_summary(test_y, y_prob) # get metrics