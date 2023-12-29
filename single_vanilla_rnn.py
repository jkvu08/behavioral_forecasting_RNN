# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet

Run single models of vanilla RNN using GRU and LSTM architecture and evaluate the results. This code is implemented within the training phase of the project.
This code can be used to manually tune parameters by monitoring the loss plots, confusion matrices and classification reports. 

"""
# Load libraries
import os
import numpy as np
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv, DataFrame
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping
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

################################
#### Model parameterization ####
################################
# generate class weights, using class weights to deal with data imbalance/skew 
weights = dict(zip([0,1,2,3], [1,1,3,1])) # create a dictionary with the weights 
features = train_X.shape[2] # get the number of features
lookback = train_X.shape[1] # get the lookback period
targets = train_y.shape[1] # get the target number
neurons_n = 10 # assign number of neurons
hidden_n = 5 # assign number of hidden neurons
d_rate = 0.3 # assign dropout rate for regularization
lr_rate = 0.001 # assign learning rate

# build model 
model = Sequential() # create an empty sequential shell 
model.add(Masking(mask_value = -1, 
                  input_shape = (lookback, features), 
                  name = 'Masking')) # add a masking layer to tell the model to ignore missing values (i.e., values of -1)
model.add(LSTM(units =neurons_n, input_shape = (lookback,features), name = 'LSTM')) # set the RNN type
model.add(Dense(units = hidden_n, activation = 'relu', kernel_initializer =  'he_uniform')) # add dense layer
model.add(Dropout(rate= d_rate)) # add dropout
model.add(Dense(units = targets, activation = "softmax", name = 'Output')) # add output layer
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = lr_rate), metrics= [F1Score(num_classes=targets, average = 'macro'),'accuracy']) # compile model, set learning rate and metrics

model.summary() # examine model architecture

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# LSTM (LSTM)                  (None, 10)                1480      
# _________________________________________________________________
# dense_3 (Dense)              (None, 5)                 55        
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 5)                 0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 24        
# =================================================================
# Total params: 1,559
# Trainable params: 1,559
# Non-trainable params: 0
# _________________________________________________________________

# fit model
history = model.fit(train_X, train_y, 
                    epochs = 50, 
                    batch_size = 512,
                    verbose = 2,
                    shuffle=False,
                    validation_data = (test_X, test_y),
                    sample_weight = None,
                    class_weight = weights)

history.history.keys()

# monitor and evaluate the results
mon_plots = bmf.monitoring_plots(history) # generate monitoring plot
mon_plots.savefig(path+'manual_vanilla_rnn_lstm_monitoring.jpg', dpi=150) # save monitoring plot
loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.455
# f1: 0.230
# accuracy: 0.852
y_val = bmf.one_hot_decode(test_y) # retrieve labels for test targets
y_prob = model.predict(test_X) # get prediction prob for each class
y_pred = bmf.to_label(y_prob,prob = True) # generate prediction labels

bmf.confusion_mat(y_val, y_pred) # generate confusion matrix 
bmf.class_report(y_val, y_pred) # generate classification report
# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
lstm_score, lstm_scores, _, _ = bmf.result_summary(test_y, y_prob, path, 'manual_vanilla_rnn_lstm') 
# view output file results

# build model using wrapper
model = bmf.build_rnn(train_X, train_y, neurons_n = 20, hidden_n = 10, lr_rate = 0.001, d_rate=0.3,layers = 1, mtype = 'GRU')
model.summary()
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                2880      
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,134
# Trainable params: 3,134
# Non-trainable params: 0
# _________________________________________________________________

# generate a callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', # monitor validation loss
                               patience = 25, # stop if loss doesn't improve after 5 epochs
                               mode = 'min', # minimize validation loss 
                               restore_best_weights=True, # restore the best weights
                               verbose=1) # print progress

# fit the model
history = model.fit(train_X, 
                    train_y, 
                    epochs = 100, 
                    batch_size = 512,
                    shuffle=False,
                    validation_data=(test_X,test_y),
                    class_weight=weights,
                    sample_weight=None,
                    callbacks = [early_stopping],
                    verbose = 2)

# early stopping activated, stopped after 26 epochs
# monitor the results
mon_plots2 = bmf.monitoring_plots(history)
mon_plots2.savefig(path+'manual_vanilla_rnn_GRU_monitoring.jpg', dpi=150) # save monitoring plot

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.466
# f1: 0.300
# accuracy: 0.856
y_prob = model.predict(test_X) # get prediction prob for each class
y_pred = bmf.to_label(y_prob,prob = True) # generate prediction labels
gru_score, gru_scores, _, _ = bmf.result_summary(test_y, y_prob, path, 'manual_vanilla_rnn_gru_evaluation')
# view output file results
