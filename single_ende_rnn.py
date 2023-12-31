# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

Run single models of encoder-decoder RNN using GRU and LSTM architecture and evaluate the results. This code is implemented within the training phase of the project.
This code can be used to manually tune parameters by monitoring the loss plots, confusion matrices and classification reports. 

@author: Jannet
"""
# Load libraries
import os
#import time
import numpy as np
from numpy import newaxis
#from matplotlib import pyplot as plt
#import pandas as pd
from pandas import read_csv
#from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
#import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, RepeatVector, TimeDistributed
#from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
#from keras.callbacks import Callback

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
# loss: 0.451
# f1: 0.229
# accuracy: 0.858

y_val = bmf.one_hot_decode(test_y) # retrieve labels for test targets
y_val[0:10] # view subset target labels
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[0.44845662, 0.39094216, 0.01825993, 0.14234132],
#        [0.15812816, 0.7079554 , 0.03350493, 0.10041149],
#        [0.21287797, 0.4884479 , 0.08845554, 0.21021867],
#        [0.11449117, 0.7072292 , 0.06240171, 0.11587794],
#        [0.06363994, 0.8338091 , 0.03948412, 0.0630669 ],
#        [0.04477714, 0.8816436 , 0.02974448, 0.04383473],
#        [0.03588417, 0.9050219 , 0.02449205, 0.03460185],
#        [0.0297618 , 0.9266335 , 0.0176416 , 0.02596321],
#        [0.03016611, 0.92537135, 0.01800338, 0.02645908],
#        [0.03028067, 0.92505103, 0.01808515, 0.02658314]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_pred[0:10] # view subset of predictions
# [1, 1, 1, 0, 1, 1, 1, 1, 1, 1] can see that 2 predictions differ from the target

cm_fig = bmf.confusion_mat(y_val, y_pred) # generate confusion matrix 
bmf.class_report(y_val, y_pred) # generate classification report

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
lstm_score, _, _, _ = bmf.result_summary(y_test, 
                                         y_prob, 
                                         path, 
                                         'manual_ende_rnn_lstm_evaluation') 
# lstm_score = 0.321
# view output file results

# build model using wrapper
model = bmf.build_ende(train_X, 
                       train_y, 
                       neurons_n = 20, 
                       hidden_n = 10, 
                       td_neurons = 10,
                       lr_rate = 0.001, 
                       d_rate=0.3,
                       layers = 1, 
                       mtype = 'GRU')

model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                2880      
# _________________________________________________________________
# dense_10 (Dense)             (None, 10)                210       
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector_3 (RepeatVecto (None, 1, 10)             0         
# _________________________________________________________________
# gru (GRU)                    (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed_6 (TimeDist (None, 1, 10)             210       
# _________________________________________________________________
# time_distributed_7 (TimeDist (None, 1, 4)              44        
# =================================================================
# Total params: 5,264
# Trainable params: 5,264
# Non-trainable params: 0
# _________________________________________________________________

# generate a callback for early stopping to prevent overfitting on training data
early_stopping = EarlyStopping(monitor='val_loss', # monitor validation loss
                               patience = 25, # stop if loss doesn't improve after 5 epochs
                               mode = 'min', # minimize validation loss 
                               restore_best_weights=True, # restore the best weights
                               verbose=1) # print progress

# fit the model
history = model.fit(train_X, # features
                    train_y, # targets
                    validation_data = (test_X, test_y), # add validation data
                    epochs = 100, # epochs 
                    batch_size = 512, # batch size
                    sample_weight = sweights, # add sample weights
                    callbacks = [early_stopping], # add early stopping callback
                    shuffle=False, # determine whether to shuffle order of data, False since we want to preserve time series
                    verbose = 2) # status print outs
# early stopping not initiated, ran 100 epochs

# monitor and evaluate the results
mon_plots2 = bmf.monitoring_plots(history, ['loss','f1','accuracy']) # generate loss and performance curves
mon_plots2.savefig(path+'manual_ende_rnn_gru_monitoring.jpg', dpi=150) # save monitoring plot and examine in output file
# loss and accuracy plots don't look great. Don't really see improvement in either.Validation f1 also doesn't seem to improve.
# may want to use f1 loss instead or run longer without early stopping

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.442
# f1: 0.229
# accuracy: 0.858

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[0.4630983 , 0.38370293, 0.02390137, 0.12929736],
#        [0.10854505, 0.78368074, 0.0121613 , 0.0956129 ],
#        [0.24994574, 0.49630412, 0.01838227, 0.23536777],
#        [0.13313739, 0.7562577 , 0.01468982, 0.09591508],
#        [0.0624603 , 0.86588496, 0.01150105, 0.06015366],
#        [0.03238265, 0.9311583 , 0.00981699, 0.02664196],
#        [0.02648979, 0.9367797 , 0.01113822, 0.02559233],
#        [0.02486929, 0.9400732 , 0.01153779, 0.02351977],
#        [0.02495077, 0.9397495 , 0.01160245, 0.02369726],
#        [0.02479382, 0.94000655, 0.01158202, 0.02361759]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [1, 1, 2, 1, 1, 1, 1, 1, 1, 1] can see that 2 predictions differ from the target, same as lstm model based on first 10 predictions 

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
gru_score, _, _, _ = bmf.result_summary(y_test, 
                                         y_prob, 
                                         path, 
                                         'manual_ende_rnn_gru_evaluation') 
# gru_score = 0.322, similar performance to lstm model

# build model using f1_loss function
model = bmf.build_ende(train_X, 
                       train_y, 
                       neurons_n = 20, 
                       hidden_n = 10, 
                       td_neurons = 10,
                       lr_rate = 0.001, 
                       d_rate=0.3,
                       layers = 1, 
                       mtype = 'GRU',
                       cat_loss = False)

model.summary()
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                2880      
# _________________________________________________________________
# dense_13 (Dense)             (None, 10)                210       
# _________________________________________________________________
# dropout_5 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector_4 (RepeatVecto (None, 1, 10)             0         
# _________________________________________________________________
# gru_1 (GRU)                  (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed_8 (TimeDist (None, 1, 10)             210       
# _________________________________________________________________
# time_distributed_9 (TimeDist (None, 1, 4)              44        
# =================================================================
# Total params: 5,264
# Trainable params: 5,264
# Non-trainable params: 0
# _________________________________________________________________

# generate a callback for early stopping to prevent overfitting on training data
early_stopping = EarlyStopping(monitor='val_loss', # monitor validation loss
                               patience = 25, # stop if loss doesn't improve after 25 epochs
                               mode = 'min', # minimize validation loss 
                               restore_best_weights=True, # restore the best weights
                               verbose=1) # print progress

# fit the model
history = model.fit(train_X, # features
                    train_y, # targets
                    validation_data = (test_X, test_y), # add validation data
                    epochs = 100, # epochs 
                    batch_size = 512, # batch size
                    sample_weight = sweights, # add sample weights
                    callbacks = [early_stopping], # add early stopping callback
                    shuffle=False, # determine whether to shuffle order of data, False since we want to preserve time series
                    verbose = 2) # status print outs

# monitor the results
mon_plots3 = bmf.monitoring_plots(history, ['loss','f1','accuracy'])
mon_plots3.savefig(path+'manual_ende_rnn_gru_monitoring_f1_loss.jpg', dpi=150) # save monitoring plot
# early stopping activated after 53 epochs
# loss and performance curves are less noisy

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# f1_loss: 0.671
# f1: 0.330
# accuracy: 0.832

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[9.9993324e-01, 1.6432096e-10, 6.6792403e-05, 3.9807942e-11],
#        [2.8437383e-12, 9.9996340e-01, 3.4641256e-05, 1.9412471e-06],
#        [3.3550171e-07, 5.4265306e-06, 4.9743128e-05, 9.9994445e-01],
#        [4.8992053e-11, 9.9964881e-01, 2.9326553e-04, 5.7916561e-05],
#        [1.1865787e-12, 9.9998343e-01, 1.5668364e-05, 9.9239060e-07],
#        [9.3730058e-13, 9.9998641e-01, 1.2724231e-05, 8.4858743e-07],
#        [9.0765617e-13, 9.9998689e-01, 1.2332683e-05, 8.3874073e-07],
#        [8.9564906e-13, 9.9998701e-01, 1.2149139e-05, 8.3680254e-07],
#        [8.8805563e-13, 9.9998713e-01, 1.2074478e-05, 8.3458065e-07],
#        [8.8093974e-13, 9.9998713e-01, 1.2001807e-05, 8.3271703e-07]],
#       dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 1, 1, 1, 1, 1, 1, 1] 3 predictions differ from target, one more incorrect han previous models based on first 10 records
gru_score_f1, _, _, _ = bmf.result_summary(y_test, 
                                           y_prob, 
                                           path, 
                                           'manual_ende_rnn_gru_evaluation_f1')
# view output file results
# gru_score_f1 = 0.442, higher f1 score compared to lstm and gru model trained using categorical cross entropy loss function
# also outperforms models trained using categorical cross entropy loss function based on overall and class specific precision, recall and accuracy based on output file

# test same model multiple times
params = {'atype': 'ENDE',
          'mtype': 'GRU',
          'hidden_layers': 1,
          'neurons_n': 20,
          'hidden_n': 10,
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

model = bmf.build_ende(train_X, 
                       train_y, 
                       layers = params['hidden_layers'], 
                       neurons_n = params['neurons_n'], 
                       hidden_n = params['hidden_n'], 
                       td_neurons = params['td_neurons'],
                       lr_rate = params['learning_rate'], 
                       d_rate= params['dropout_rate'],
                       mtype = params['mtype'], 
                       cat_loss = params['loss'])

model.summary()
# Model: "sequential_3"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                2880      
# _________________________________________________________________
# dense_10 (Dense)             (None, 10)                210       
# _________________________________________________________________
# dropout_10 (Dropout)         (None, 10)                0         
# _________________________________________________________________
# repeat_vector (RepeatVector) (None, 1, 10)             0         
# _________________________________________________________________
# gru (GRU)                    (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed (TimeDistri (None, 1, 5)              105       
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, 1, 4)              24        
# =================================================================
# Total params: 5,139
# Trainable params: 5,139
# Non-trainable params: 0
# _________________________________________________________________

# run without early stopping, 5 trials
eval_tab, avg_eval = bmf.eval_iter(model, 
                                   params, 
                                   train_X, 
                                   train_y, 
                                   test_X, 
                                   test_y, 
                                   patience = 0, 
                                   max_epochs = params['epochs'], 
                                   atype = params['atype'], 
                                   n =5)

eval_tab # epochs run, loss and metrics at the end of each model iteration 
#    epochs      loss        f1  accuracy  val_loss    val_f1  val_accuracy
# 0     100  0.607589  0.407723  0.778060  0.797386  0.201555      0.744683
# 1     100  0.587712  0.426673  0.773860  0.577400  0.421629      0.840037
# 2     100  0.585749  0.429329  0.774739  0.580143  0.418816      0.839465
# 3     100  0.580732  0.433705  0.777498  0.580151  0.419006      0.840836
# 4     100  0.579447  0.434846  0.778548  0.582185  0.417092      0.839922
# similar performance across iterations except for the first iteration

avg_eval # average epochs run, loss and metrics
# epochs          100.000000
# loss              0.588246
# f1                0.426455
# accuracy          0.776541
# val_loss          0.623453
# val_f1            0.375620
# val_accuracy      0.820989
# dtype: float64

# run with early stopping with patience = 50, stopped val loss does not improve for 50 epochs
eval_tab, avg_eval = bmf.eval_iter(model, 
                                   params, 
                                   train_X, 
                                   train_y, 
                                   test_X, 
                                   test_y, 
                                   patience = 50, 
                                   max_epochs = params['epochs'], 
                                   atype = params['atype'], 
                                   n = 5)

eval_tab # epochs run, loss and metrics at the end of each model iteration 
#    epochs      loss        f1  accuracy  val_loss    val_f1  val_accuracy
# 0      21  0.580059  0.434260  0.777816  0.578979  0.420193      0.839751
# 1       6  0.578403  0.435985  0.778817  0.579158  0.420503      0.840722
# 2     100  0.581006  0.433445  0.778084  0.589150  0.410152      0.839808
# 3       9  0.579249  0.435162  0.779330  0.578513  0.419461      0.841465
# 4     100  0.578671  0.436076  0.778646  0.579156  0.419579      0.839808
# variation in epochs run, however loss and metrics were consistent between runs

avg_eval # average epochs run, loss and metrics
# epochs          47.200000
# loss             0.579477
# f1               0.434985
# accuracy         0.778538
# val_loss         0.580991
# val_f1           0.417977
# val_accuracy     0.840311
# dtype: float64
# similar metrics as the run without patience, likely cause loss and metrics plateaued
# seems like early stopping can be applied given the similar performances of the models despite running for various epochs