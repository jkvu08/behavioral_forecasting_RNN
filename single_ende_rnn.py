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
from tensorflow_addons.metrics import F1Score
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

# copy over 2d test set, will be used to evaluate later
y_test = np.copy(test_y)

# format for encoder-decoder model 
if n_output == 1:
    test_y = test_y[:,newaxis,:]
    train_y = train_y[:,newaxis,:]
    
################################
#### Model parameterization ####
################################
# generate class weights, using class weights to deal with data imbalance/skew 
weights = dict(zip([0,1,2,3], [1,1,3,1])) # create a dictionary with the weights 
sweights = bmf.get_sample_weights(train_y, weights)

lookback = train_X.shape[1] # set lookback
features = train_X.shape[2] # set features
targets = train_y.shape[2] # set number of targets per timesteps

neurons_n = 10 # assign number of neurons
hidden_n = 10 # assign number of hidden neurons
td_neurons = 5 # assign number of time distributed neurons
d_rate = 0.3 # assign dropout rate for regularization
lr_rate = 0.001 # assign learning rate

# build model 
model = Sequential() # create an empty sequential shell 
# add a masking layer to tell the model to ignore missing values (i.e., values of -1, since that was used to designate missing values)
model.add(Masking(mask_value = -1, 
                  input_shape = (lookback, features), 
                  name = 'Masking')) 
# set the RNN type
model.add(LSTM(units =neurons_n, 
               input_shape = (lookback,features), 
               name = 'LSTM')) 
# add dense layer & set activation function
model.add(Dense(units = hidden_n, 
                activation = 'relu', 
                kernel_initializer =  'he_uniform')) 
# add dropout
model.add(Dropout(rate= d_rate)) 
# repeats encoder context for each prediction timestep
model.add(RepeatVector(n_output)) 
# add approriate RNN type after repeat vector
model.add(LSTM(units = neurons_n, 
               input_shape = (lookback,features), 
               return_sequences=True)) 
# make sequential predictions, applies decoder fully connected layer to each prediction timestep
model.add(TimeDistributed(Dense(units = td_neurons, activation='relu')))
# applies output layer to each prediction timestep
model.add(TimeDistributed(Dense(targets, activation = "softmax"))) 
# compile model 
model.compile(loss = 'categorical_crossentropy', # use categorical crossentropy loss
              optimizer = Adam(learning_rate = lr_rate), # set learning rate 
              metrics = [bmf.f1,'accuracy'], # monitor metrics
              sample_weight_mode = 'temporal') # add sample weights, since class weights are not supported in 3D

model.summary() # examine model architecture
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# LSTM (LSTM)                  (None, 10)                1480      
# _________________________________________________________________
# dense (Dense)                (None, 10)                110       
# _________________________________________________________________
# dropout (Dropout)            (None, 10)                0         
# _________________________________________________________________
# repeat_vector (RepeatVector) (None, 1, 10)             0         
# _________________________________________________________________
# lstm (LSTM)                  (None, 1, 10)             840       
# _________________________________________________________________
# time_distributed (TimeDistri (None, 1, 5)              55        
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, 1, 4)              24        
# =================================================================
# Total params: 2,509
# Trainable params: 2,509
# Non-trainable params: 0
# _________________________________________________________________

# fit model
history = model.fit(train_X, # features
                    train_y, # targets
                    validation_data = (test_X, test_y), # add validation data
                    epochs = 50, # epochs 
                    batch_size = 512, # batch size
                    sample_weight = sweights, # add sample weights
                    shuffle=False, # determine whether to shuffle order of data, False since we want to preserve time series
                    verbose = 2) # status print outs

history.history.keys() # examine outputs
# dict_keys(['loss', 'f1', 'accuracy', 'val_loss', 'val_f1', 'val_accuracy'])

# monitor and evaluate the results
mon_plots = bmf.monitoring_plots(history, ['loss','f1','accuracy']) # generate loss and performance curves
mon_plots.savefig(path+'manual_ende_rnn_lstm_monitoring.jpg', dpi=150) # save monitoring plot and examine in output file
# 
loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.445
# f1: 0.229
# accuracy: 0.858

y_val = bmf.one_hot_decode(test_y) # retrieve labels for test targets
y_val[0:10] # view subset target labels
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[0.46956554, 0.3628467 , 0.02484568, 0.142742  ],
#        [0.16406503, 0.7005112 , 0.03209489, 0.10332887],
#        [0.18590716, 0.491183  , 0.06099619, 0.26191372],
#        [0.1086543 , 0.7303062 , 0.04109134, 0.11994815],
#        [0.06124726, 0.85508233, 0.02818266, 0.05548776],
#        [0.04101677, 0.8982838 , 0.02270741, 0.03799199],
#        [0.03531985, 0.9124129 , 0.01978673, 0.03248053],
#        [0.02661475, 0.9351574 , 0.01411079, 0.02411705],
#        [0.02724199, 0.93386066, 0.01441905, 0.02447829],
#        [0.02744985, 0.93352395, 0.01450501, 0.0245211 ]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob,
                      prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_pred[0:10] # view subset of predictions
# [3, 0, 1, 1, 1, 1, 1, 1, 1, 1] can see that 2 predictions differ from the target

cm_fig = bmf.confusion_mat(y_val, y_pred) # generate confusion matrix 
bmf.class_report(y_val, y_pred) # generate classification report

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
lstm_score, _, _, _ = bmf.result_summary(y_test, 
                                         y_prob, 
                                         path, 
                                         'manual_ende_rnn_lstm_evaluation') 
# lstm_score = 0.325
# view output file results