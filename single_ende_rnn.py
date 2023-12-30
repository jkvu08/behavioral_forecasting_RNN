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
#from matplotlib import pyplot
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
# early stopping initiated at 26 epochs

# monitor and evaluate the results
mon_plots2 = bmf.monitoring_plots(history, ['loss','f1','accuracy']) # generate loss and performance curves
mon_plots2.savefig(path+'manual_ende_rnn_gru_monitoring.jpg', dpi=150) # save monitoring plot and examine in output file
# loss and accuracy plots don't look great. Don't really see improvement in either.Validation f1 also doesn't seem to improve.
# may want to use f1 loss instead or run longer without early stopping

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.451
# f1: 0.229
# accuracy: 0.858

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[0.40441394, 0.36675656, 0.02048454, 0.2083449 ],
#        [0.10962416, 0.76384246, 0.00764013, 0.11889327],
#        [0.16252732, 0.6288581 , 0.01860255, 0.19001201],
#        [0.09531359, 0.7889532 , 0.00916176, 0.10657147],
#        [0.05646299, 0.8538656 , 0.01055881, 0.07911263],
#        [0.04969491, 0.8681456 , 0.01146916, 0.07069026],
#        [0.04442912, 0.87876123, 0.01473773, 0.06207189],
#        [0.03180532, 0.9160052 , 0.01597678, 0.0362127 ],
#        [0.03196592, 0.9153372 , 0.01580463, 0.03689224],
#        [0.03183167, 0.9157397 , 0.01571081, 0.03671776]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob,
                      prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 1, 1, 1, 1, 1, 1, 1, 0] can see that 3 predictions differ from the target, more inaccurate than lstm based on first 10 predictions

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
# early stopping activated after 87 epochs
# loss and performance curves are less noisy

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# f1_loss: 0.669
# f1: 0.331
# accuracy: 0.839

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[1.0000000e+00, 7.4371403e-12, 7.0029212e-11, 1.0982383e-09],
#        [5.5838521e-15, 9.9999213e-01, 7.8826079e-06, 2.5520348e-12],
#        [3.0090220e-08, 4.4974400e-16, 5.1900759e-07, 9.9999940e-01],
#        [6.2020519e-15, 9.9999011e-01, 9.9299323e-06, 3.3750717e-12],
#        [5.4428444e-15, 9.9999261e-01, 7.4382433e-06, 2.3721259e-12],
#        [5.5248020e-15, 9.9999225e-01, 7.7185423e-06, 2.4777149e-12],
#        [5.5633353e-15, 9.9999225e-01, 7.7366940e-06, 2.4896385e-12],
#        [5.5706617e-15, 9.9999225e-01, 7.7368713e-06, 2.4907643e-12],
#        [5.5758070e-15, 9.9999225e-01, 7.7471414e-06, 2.4950007e-12],
#        [5.5810205e-15, 9.9999225e-01, 7.7578170e-06, 2.4993924e-12]],
#       dtype=float32)

# ensure row probabilities equal to 1, might slightly deviate due to approximation of f1_loss function
# subtract a small amount from the largest class probability per row
y_proba = bmf.prob_adjust(y_prob)
y_proba[0:10,:]
# array([[9.9998999e-01, 7.4371403e-12, 7.0029212e-11, 1.0982383e-09],
#        [5.5838521e-15s, 9.9998212e-01, 7.8826079e-06, 2.5520348e-12],
#        [3.0090220e-08, 4.4974400e-16, 5.1900759e-07, 9.9998939e-01],
#        [6.2020519e-15, 9.9998009e-01, 9.9299323e-06, 3.3750717e-12],
#        [5.4428444e-15, 9.9998260e-01, 7.4382433e-06, 2.3721259e-12],
#        [5.5248020e-15, 9.9998224e-01, 7.7185423e-06, 2.4777149e-12],
#        [5.5633353e-15, 9.9998224e-01, 7.7366940e-06, 2.4896385e-12],
#        [5.5706617e-15, 9.9998224e-01, 7.7368713e-06, 2.4907643e-12],
#        [5.5758070e-15, 9.9998224e-01, 7.7471414e-06, 2.4950007e-12],
#        [5.5810205e-15, 9.9998224e-01, 7.7578170e-06, 2.4993924e-12]],
#       dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_proba,
                      prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 1, 1, 1, 1, 1, 1, 1] 3 predictions differ from target, same performance of previous gru model based on accuracy of first 10 records
gru_score_f1, _, _, _ = bmf.result_summary(y_test, 
                                           y_prob, 
                                           path, 
                                           'manual_ende_rnn_gru_evaluation_f1')
# view output file results
# gru_score_f1 = 0.444, higher f1 score compared to lstm and gru model trained using categorical cross entropy loss function
# also outperforms models trained using categorical cross entropy loss function based on overall and class specific precision, recall and accuracy based on output file