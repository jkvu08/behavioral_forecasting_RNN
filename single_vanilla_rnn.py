# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet

Run single models of vanilla RNN using GRU and LSTM architecture and evaluate the results. This code is implemented within the training phase of the project.
This code can be used to manually tune parameters by monitoring the loss plots, confusion matrices and classification reports. 

"""
# Load libraries
import os
import pandas as pd
from pandas import read_csv
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping

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
# add output layer
model.add(Dense(units = targets, 
                activation = "softmax", 
                name = 'Output')) 
model.compile(loss = 'categorical_crossentropy', # compile model, 
              optimizer = Adam(learning_rate = lr_rate), # set learning rate
              metrics= [F1Score(num_classes=targets, average = 'macro'),'accuracy']) # calculate metrics

model.summary() # examine model architecture

# Model: "sequential"
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
history = model.fit(train_X, # features
                    train_y, # targets
                    validation_data = (test_X, test_y), # add validation data
                    epochs = 50, # epochs 
                    batch_size = 512, # batch size
                  #  class_weight = weights, # add class weights
                    shuffle=False, # determine whether to shuffle order of data, False since we want to preserve time series
                    verbose = 2) # status print outs
history.history.keys() # examine outputs
# dict_keys(['loss', 'f1', 'accuracy', 'val_loss', 'val_f1', 'val_accuracy'])

# monitor and evaluate the results
mon_plots = bmf.monitoring_plots(history, ['loss','f1','accuracy']) # generate loss and performance curves
mon_plots.savefig(path+'manual_vanilla_rnn_lstm_monitoring.jpg', dpi=150) # save monitoring plot and examine in output file
# signs of overfit to training, might want to add early stopping

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.445
# f1: 0.246
# accuracy: 0.853

y_val = bmf.one_hot_decode(test_y) # retrieve labels for test targets
y_val[0:10] # view subset target labels
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob[0:10,:] # print subset
# array([[0.38152137, 0.40610015, 0.0292882 , 0.18309031],
#        [0.1326671 , 0.7278745 , 0.0394221 , 0.10003637],
#        [0.21834752, 0.559865  , 0.05685439, 0.16493316],
#        [0.08689535, 0.8121106 , 0.03092397, 0.07007006],
#        [0.04525043, 0.8948455 , 0.02044962, 0.03945447],
#        [0.03561175, 0.9164246 , 0.01671387, 0.03124982],
#        [0.03409444, 0.9203952 , 0.01586975, 0.0296407 ],
#        [0.02601318, 0.94270724, 0.01092626, 0.02035322],
#        [0.02676555, 0.94088787, 0.01129478, 0.02105178],
#        [0.02675696, 0.9409276 , 0.01128296, 0.02103235]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob,
                      prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_pred[0:10] # view subset of predictions
# [1, 1, 0, 1, 1, 1, 1, 0, 3, 1] can see that 4 predictions differ from the target

cm_fig = bmf.confusion_mat(y_val, y_pred) # generate confusion matrix 
bmf.class_report(y_val, y_pred) # generate classification report

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
lstm_score, _, _, _ = bmf.result_summary(test_y, 
                                         y_prob, 
                                         path, 
                                         'manual_vanilla_rnn_lstm_evaluation') 
# lstm_score = 0.316

# view output file results

# build model using wrapper
model = bmf.build_rnn(train_X, 
                      train_y, 
                      neurons_n = 20, 
                      hidden_n = 10, 
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

# generate a callback for early stopping to prevent overfitting on training data
early_stopping = EarlyStopping(monitor='val_loss', # monitor validation loss
                               patience = 25, # stop if loss doesn't improve after 5 epochs
                               mode = 'min', # minimize validation loss 
                               restore_best_weights=True, # restore the best weights
                               verbose=1) # print progress

# fit the model
history = model.fit(train_X, 
                    train_y, 
                    validation_data=(test_X,test_y),
                    epochs = 100, 
                    batch_size = 512,
                    class_weight=weights,
                #    callbacks = [early_stopping],
                    shuffle=False,
                    verbose = 2)

# early stopping activated, stopped after 61 epochs
# monitor the results
mon_plots2 = bmf.monitoring_plots(history, ['loss','f1','accuracy'])
mon_plots2.savefig(path+'manual_vanilla_rnn_gru_monitoring.jpg', dpi=150) # save monitoring plot
# loss and acc plateaus, f1 still oscillating
# could try running with f1 loss function instead

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.444
# f1: 0.285
# accuracy: 0.855
y_prob = model.predict(test_X) # get prediction prob for each class
y_prob[0:10,:] # print subset
# array([[0.39005977, 0.3974965 , 0.01779572, 0.19464797],
#        [0.14086   , 0.7013224 , 0.03334076, 0.12447689],
#        [0.19043007, 0.5399154 , 0.04155972, 0.22809483],
#        [0.09804849, 0.7587881 , 0.03228288, 0.11088052],
#        [0.05839545, 0.85624164, 0.02321226, 0.06215064],
#        [0.03911531, 0.9043857 , 0.01729717, 0.03920171],
#        [0.02995602, 0.92646945, 0.01467866, 0.02889583],
#        [0.021499  , 0.94925356, 0.01104413, 0.01820329],
#        [0.02160019, 0.9488647 , 0.01116069, 0.01837451],
#        [0.02142691, 0.9492494 , 0.01109311, 0.01823056]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob,
                      prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [1, 3, 3, 1, 0, 1, 3, 1, 1, 1] 3 predictions differ from target, slightly better alignment than previous lstm model based accuracy of first 10 records
gru_score, _, _, _ = bmf.result_summary(test_y, 
                                        y_prob, 
                                        path, 
                                        'manual_vanilla_rnn_gru_evaluation')
# view output file results
# gru_score = 0.327, slightly higher f1 score compared to lstm model

# build model using f1_loss function
model = bmf.build_rnn(train_X, 
                      train_y, 
                      neurons_n = 20, 
                      hidden_n = 10, 
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

# generate a callback for early stopping to prevent overfitting on training data
early_stopping = EarlyStopping(monitor='val_loss', # monitor validation loss
                               patience = 25, # stop if loss doesn't improve after 25 epochs
                               mode = 'min', # minimize validation loss 
                               restore_best_weights=True, # restore the best weights
                               verbose=1) # print progress

# fit the model
history1 = model.fit(train_X, 
                    train_y, 
                    validation_data=(test_X,test_y),
                    epochs = 100, 
                    batch_size = 512,
                    class_weight=weights,
                    callbacks = [early_stopping],
                    shuffle=False,
                    verbose = 2)

# monitor the results
mon_plots3 = bmf.monitoring_plots(history, ['loss','f1','accuracy'])
mon_plots3.savefig(path+'manual_vanilla_rnn_gru_monitoring_f1_loss.jpg', dpi=150) # save monitoring plot
# loss and performance curves are less noisy
# early stopping activated after 95 epochs
# 100 epochs seems reasonable then

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# f1_loss: 0.667
# f1: 0.443
# accuracy: 0.837

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob[0:10,:] # print subset
# array([[1.00000000e+00, 2.08646195e-15, 1.93955227e-10, 3.30982013e-11],
#        [9.09008349e-07, 9.98719096e-01, 1.41719750e-06, 1.27853698e-03],
#        [8.69106564e-09, 2.24256985e-12, 1.26300952e-08, 1.00000000e+00],
#        [5.29656180e-13, 9.99999881e-01, 1.36456059e-12, 9.39466887e-08],
#        [4.14612966e-17, 1.00000000e+00, 2.84190756e-15, 1.92606694e-10],
#        [1.31171382e-17, 1.00000000e+00, 3.07579008e-15, 7.30266611e-11],
#        [1.54940789e-17, 1.00000000e+00, 4.12304341e-15, 7.54732110e-11],
#        [3.16392217e-17, 1.00000000e+00, 9.27569856e-15, 1.07979320e-10],
#        [3.15573771e-17, 1.00000000e+00, 9.38260344e-15, 1.07487796e-10],
#        [2.93278772e-17, 1.00000000e+00, 8.81333292e-15, 1.02930872e-10]],
#       dtype=float32)

# ensure row probabilities equal to 1, might slightly deviate due to approximation of f1_loss function
# subtract a small amount from the largest class probability per row
y_proba = bmf.prob_adjust(y_prob)
y_proba[0:10,:]
# array([[9.99989986e-01, 2.08646195e-15, 1.93955227e-10, 3.30982013e-11],
#        [9.09008349e-07, 9.98709083e-01, 1.41719750e-06, 1.27853698e-03],
#        [8.69106564e-09, 2.24256985e-12, 1.26300952e-08, 9.99989986e-01],
#        [5.29656180e-13, 9.99989867e-01, 1.36456059e-12, 9.39466887e-08],
#        [4.14612966e-17, 9.99989986e-01, 2.84190756e-15, 1.92606694e-10],
#        [1.31171382e-17, 9.99989986e-01, 3.07579008e-15, 7.30266611e-11],
#        [1.54940789e-17, 9.99989986e-01, 4.12304341e-15, 7.54732110e-11],
#        [3.16392217e-17, 9.99989986e-01, 9.27569856e-15, 1.07979320e-10],
#        [3.15573771e-17, 9.99989986e-01, 9.38260344e-15, 1.07487796e-10],
#        [2.93278772e-17, 9.99989986e-01, 8.81333292e-15, 1.02930872e-10]],
#       dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_proba,
                      prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 1, 1, 1, 1, 1, 1, 1] 3 predictions differ from target, same performance of previous gru model based on accuracy of first 10 records
gru_score_f1, _, _, _ = bmf.result_summary(test_y, 
                                           y_prob, 
                                           path, 
                                           'manual_vanilla_rnn_gru_evaluation_f1')
# view output file results
# gru_score_f1 = 0.442, higher f1 score compared to lstm and gru model trained using categorical cross entropy loss function
# also outperforms models trained using categorical cross entropy loss function based on overall and class specific precision, recall and accuracy

# test same model multiple times
params = {'atype': 'VRNN',
          'mtype': 'GRU',
          'hidden_layers': 1,
          'neurons_n': 20,
          'hidden_n': 10,
          'learning_rate': 0.001,
          'dropout_rate': 0.3,               
          'loss': False,
          'epochs': 100,
          'batch_size': 512,
          'weights_0': 1,
          'weights_1': 1,
          'weights_2': 3,
          'weights_3': 1}

model = bmf.build_rnn(train_X, 
                      train_y, 
                      layers = params['hidden_layers'], 
                      neurons_n = params['neurons_n'], 
                      hidden_n = params['hidden_n'], 
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
# dense_6 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_6 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,134
# Trainable params: 3,134
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
# 0     100  0.572928  0.442878  0.772688  0.580334  0.418397      0.838323
# 1     100  0.569342  0.445388  0.774495  0.580385  0.418306      0.838665
# 2     100  0.566893  0.447899  0.775716  0.581640  0.417230      0.837751
# 3     100  0.564228  0.449986  0.776253  0.580436  0.418522      0.836952
# 4     100  0.565326  0.449062  0.775545  0.584795  0.413532      0.833295
# similar performance

avg_eval # average epochs run, loss and metrics
# epochs          100.000000
# loss              0.567744
# f1                0.447043
# accuracy          0.774940
# val_loss          0.581518
# val_f1            0.417197
# val_accuracy      0.836997
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
# 0      29  0.563981  0.450507  0.776888  0.580384  0.418536      0.838494
# 1     100  0.561938  0.452078  0.777376  0.584681  0.414455      0.836152
# 2      31  0.561969  0.452241  0.778011  0.581433  0.417510      0.839865
# 3       1  0.562095  0.452232  0.777889  0.580416  0.418613      0.839865
# 4     100  0.559126  0.455184  0.779012  0.583763  0.414756      0.837866
# variation in epochs run, however loss and metrics were consistent between runs

avg_eval # average epochs run, loss and metrics
# epochs          52.200000
# loss             0.561822
# f1               0.452449
# accuracy         0.777835
# val_loss         0.582136
# val_f1           0.416774
# val_accuracy     0.838448
# dtype: float64
# similar metrics as the run without patience, likely cause loss and metrics plateaued
