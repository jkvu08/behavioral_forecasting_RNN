# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

Run single models of encoder-decoder RNN using GRU and LSTM architecture and evaluate the results. This code is implemented within the training phase of the project.
This code can be used to manually tune parameters by monitoring the loss plots, confusion matrices and classification reports. 

@author: Jannet
"""
# Load libraries
import os
import numpy as np
from numpy import newaxis
from pandas import read_csv
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

######################
#### Data Import #####
######################
# set working directory
os.chdir("\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\behavioral_forecasting_RNN\\outputs\\"

# import packages from file
import behavior_model_func as bmf

# import training datafiles
datasub =  read_csv('kian_trainset_focal.csv', 
                    header = 0, 
                    index_col = 0)

# get rid of these predictors since they are sparse or don't see relevant 
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
                   'since_rest', 'since_feed', 'since_travel', 'sex', # internal features  
                   'gestation', 'lactation', 'mating', 'nonreproductive', # internal or external features - reproductive state can be both
                   'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count', # external features
                   'years', 'minutes_sin', 'minutes_cos', 'doy_sin','doy_cos', # external/time features
                   'adults', 'infants', 'juveniles', # external/group features
                   'individual_continuity', 'length', 'position']] # sampling features 

#####################
#### Data Format ####
#####################
# assign input and output 
n_input = 5
n_output = 1

# split dataset
train, test = bmf.split_dataset(datasub, 2015)

# format the training and testing data for the model
train_X, train_y, train_dft = bmf.to_supervised(data = train.iloc[:,6:34], 
                                                TID = train['TID'], 
                                                window = 1, 
                                                lookback = n_input, 
                                                n_output= n_output)
     
test_X, test_y, test_dft = bmf.to_supervised(data = test.iloc[:,6:34], 
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
# add dropout
model.add(Dropout(rate= d_rate)) 
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
# Masking (Masking)            (None, 5, 28)             0         
# _________________________________________________________________
# LSTM (LSTM)                  (None, 10)                1560      
# _________________________________________________________________
# dropout (Dropout)            (None, 10)                0         
# _________________________________________________________________
# dense (Dense)                (None, 10)                110       
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector (RepeatVector) (None, 1, 10)             0         
# _________________________________________________________________
# lstm (LSTM)                  (None, 1, 10)             840       
# _________________________________________________________________
# time_distributed (TimeDistri (None, 1, 5)              55        
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, 1, 4)              24        
# =================================================================
# Total params: 2,589
# Trainable params: 2,589
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
# loss: 0.455
# f1: 0.229
# accuracy: 0.857

y_val = bmf.one_hot_decode(test_y) # retrieve labels for test targets
y_val[0:10] # view subset target labels
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[0.36814365, 0.37501162, 0.04405799, 0.21278673],
#        [0.16959274, 0.65898234, 0.03657259, 0.1348523 ],
#        [0.3421596 , 0.40157267, 0.03306411, 0.22320363],
#        [0.16117309, 0.67204934, 0.03303935, 0.13373825],
#        [0.06365227, 0.8443797 , 0.02706802, 0.06489995],
#        [0.04267988, 0.8860047 , 0.02437405, 0.04694144],
#        [0.03347618, 0.9056271 , 0.02194544, 0.03895131],
#        [0.02785301, 0.91813916, 0.02017465, 0.03383319],
#        [0.02792138, 0.9179516 , 0.02026778, 0.03385924],
#        [0.02791182, 0.91795594, 0.02030193, 0.03383042]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 1, 1, 1, 1, 1, 1, 1] can see that 3 predictions differ from the target of the first 10 records

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
model = bmf.build_ende(features,
                       targets,
                       lookback,
                       n_output,
                       neurons_n0 = 20,
                       neurons_n1 = 20,
                       hidden_n = [10], 
                       td_neurons = 10,
                       learning_rate = 0.001, 
                       dropout_rate=0.3,
                       layers = 1, 
                       mtype = 'GRU')

model.summary()
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
# dense_3 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector_1 (RepeatVecto (None, 1, 10)             0         
# _________________________________________________________________
# gru (GRU)                    (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, 1, 10)             210       
# _________________________________________________________________
# time_distributed_3 (TimeDist (None, 1, 4)              44        
# =================================================================
# Total params: 5,384
# Trainable params: 5,384
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
# early stopping initiated at 85 epochs

# monitor and evaluate the results
mon_plots2 = bmf.monitoring_plots(history, ['loss','f1','accuracy']) # generate loss and performance curves
mon_plots2.savefig(path+'manual_ende_rnn_gru_monitoring.jpg', dpi=150) # save monitoring plot and examine in output file
# loss and accuracy plots don't look great. Don't really see improvement in either.Validation f1 also doesn't seem to improve.
# may want to use f1 loss instead or run longer without early stopping

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.446
# f1: 0.229
# accuracy: 0.852

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[0.40245047, 0.4382511 , 0.02835232, 0.13094606],
#        [0.13905121, 0.6623024 , 0.04592734, 0.15271908],
#        [0.22376238, 0.4660406 , 0.05316788, 0.25702912],
#        [0.10624872, 0.7478872 , 0.03745273, 0.10841131],
#        [0.059898  , 0.858211  , 0.02436849, 0.05752257],
#        [0.04104731, 0.8998216 , 0.01944244, 0.0396887 ],
#        [0.03652142, 0.90997785, 0.01853381, 0.03496693],
#        [0.03005234, 0.9250387 , 0.01742711, 0.02748181],
#        [0.03009958, 0.92495096, 0.01740921, 0.0275402 ],
#        [0.03003017, 0.9251544 , 0.01735078, 0.02746467]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 0, 0, 1, 0, 1, 3, 1, 1, 1] can see that 5 predictions differ from the target, worse than lstm model based on first 10 predictions 

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
gru_score, _, _, _ = bmf.result_summary(y_test, 
                                         y_prob, 
                                         path, 
                                         'manual_ende_rnn_gru_evaluation') 
# gru_score = 0.325, similar performance to lstm model

# build model using f1_loss function
model = bmf.build_ende(features,
                       targets,
                       lookback,
                       n_output,
                       neurons_n0 = 20,
                       neurons_n1 = 20,
                       hidden_n = [10], 
                       td_neurons = 10,
                       learning_rate = 0.001, 
                       dropout_rate=0.3,
                       layers = 1, 
                       mtype = 'GRU',
                       cat_loss = False)

model.summary()
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 28)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                3000      
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 20)                0         
# _________________________________________________________________
# dense_6 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_5 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector_2 (RepeatVecto (None, 1, 10)             0         
# _________________________________________________________________
# gru_1 (GRU)                  (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed_4 (TimeDist (None, 1, 10)             210       
# _________________________________________________________________
# time_distributed_5 (TimeDist (None, 1, 4)              44        
# =================================================================
# Total params: 5,384
# Trainable params: 5,384
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
# early stopping activated after 75 epochs
# loss and performance curves are less noisy

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# f1_loss: 0.671
# f1: 0.330
# accuracy: 0.832

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[9.99997497e-01, 6.48366605e-09, 1.78888339e-12, 2.50428980e-06],
#        [1.47256457e-10, 9.99999881e-01, 1.28709928e-07, 1.97904920e-16],
#        [6.11202267e-05, 1.08714971e-06, 2.25291253e-04, 9.99712527e-01],
#        [1.39887074e-10, 9.99999881e-01, 1.13502509e-07, 1.64826748e-16],
#        [1.24546595e-10, 9.99999881e-01, 1.12923800e-07, 1.62741234e-16],
#        [1.28978453e-10, 9.99999881e-01, 1.14196624e-07, 1.64506385e-16],
#        [1.29721317e-10, 9.99999881e-01, 1.14911103e-07, 1.66255164e-16],
#        [1.31972142e-10, 9.99999881e-01, 1.16588744e-07, 1.70067977e-16],
#        [1.31655603e-10, 9.99999881e-01, 1.16442294e-07, 1.69783413e-16],
#        [1.31364641e-10, 9.99999881e-01, 1.16313437e-07, 1.69535537e-16]],
#       dtype=float32

# ensure row probabilities equal to 1, might slightly deviate due to approximation of f1_loss function
# subtract a small amount from the largest class probability per row
y_proba = bmf.prob_adjust(y_prob)
y_proba[0:10,:]
# array([[9.99997377e-01, 6.48366605e-09, 1.78888339e-12, 2.50428980e-06],
#        [1.47256457e-10, 9.99999762e-01, 1.28709928e-07, 1.97904920e-16],
#        [6.11202267e-05, 1.08714971e-06, 2.25291253e-04, 9.99712408e-01],
#        [1.39887074e-10, 9.99999762e-01, 1.13502509e-07, 1.64826748e-16],
#        [1.24546595e-10, 9.99999762e-01, 1.12923800e-07, 1.62741234e-16],
#        [1.28978453e-10, 9.99999762e-01, 1.14196624e-07, 1.64506385e-16],
#        [1.29721317e-10, 9.99999762e-01, 1.14911103e-07, 1.66255164e-16],
#        [1.31972142e-10, 9.99999762e-01, 1.16588744e-07, 1.70067977e-16],
#        [1.31655603e-10, 9.99999762e-01, 1.16442294e-07, 1.69783413e-16],
#        [1.31364641e-10, 9.99999762e-01, 1.16313437e-07, 1.69535537e-16]],
#       dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_proba, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 1, 1, 1, 1, 1, 1, 1] 3 predictions differ from target, more accurate compared to previous models based on first 10 records
gru_score_f1, _, _, _ = bmf.result_summary(y_test, 
                                           y_prob, 
                                           path, 
                                           'manual_ende_rnn_gru_evaluation_f1')
# view output file results
# gru_score_f1 = 0.443, higher f1 score compared to lstm and gru model trained using categorical cross entropy loss function
# also outperforms models trained using categorical cross entropy loss function based on overall and class specific precision, recall and accuracy based on output file

# test same model multiple times
params = {'atype': 'ENDE',
          'mtype': 'GRU',
          'hidden_layers': 1,
          'neurons_n0': 20,
          'neurons_n1': 20,
          'hidden_n': 10,
          'td_neurons': 5,
          'learning_rate': 0.001,
          'dropout_rate': 0.3,               
          'loss': False,
          'epochs': 100,
          'max_epochs':100,
          'batch_size': 512,
          'weights_0': 1,
          'weights_1': 1,
          'weights_2': 3,
          'weights_3': 1}


model = bmf.build_ende(features,
                       targets,
                       lookback,
                       n_output, 
                       layers = params['hidden_layers'], 
                       neurons_n0 = params['neurons_n0'],
                       neurons_n1 = params['neurons_n1'],
                       hidden_n=[params['hidden_n']],
                       td_neurons = params['td_neurons'],
                       learning_rate = params['learning_rate'], 
                       dropout_rate= params['dropout_rate'],
                       mtype = params['mtype'], 
                       cat_loss = params['loss'])

model.summary()
# Model: "sequential_3"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 28)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                3000      
# _________________________________________________________________
# dropout_6 (Dropout)          (None, 20)                0         
# _________________________________________________________________
# dense_9 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_7 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# repeat_vector_3 (RepeatVecto (None, 1, 10)             0         
# _________________________________________________________________
# gru_2 (GRU)                  (None, 1, 20)             1920      
# _________________________________________________________________
# time_distributed_6 (TimeDist (None, 1, 5)              105       
# _________________________________________________________________
# time_distributed_7 (TimeDist (None, 1, 4)              24        
# =================================================================
# Total params: 5,259
# Trainable params: 5,259
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
                                   max_epochs = params['max_epochs'], 
                                   atype = params['atype'], 
                                   n = 5)

eval_tab # epochs run, loss and metrics at the end of each model iteration 
#    epochs  iter train_loss  train_f1  train_acc  val_loss    val_f1   val_acc
# 0     100     0   0.608447  0.406829   0.769539  0.586793  0.412326  0.838151
# 1     100     1   0.605373  0.410112   0.760188  0.587452  0.412080  0.830953
# 2     100     2   0.591148  0.423348   0.767488  0.578478  0.419921  0.834438
# 3     100     3   0.588404  0.427115   0.772102  0.586721  0.411934  0.837523
# 4     100     4   0.587500  0.427543   0.769441  0.587911  0.410522  0.835809
# similar performance across iterations, may not need to test with multiple iterations

avg_eval # average epochs run, loss and metrics
# epochs        100.000000
# train_loss      0.596174
# train_f1        0.418989
# train_acc       0.767752
# val_loss        0.585471
# val_f1          0.413357
# val_acc         0.835375
# dtype: float64

# run with early stopping with patience = 50, stopped val loss does not improve for 50 epochs
eval_tab, avg_eval = bmf.eval_iter(model, 
                                   params, 
                                   train_X, 
                                   train_y, 
                                   test_X, 
                                   test_y, 
                                   patience = 50, 
                                   max_epochs = params['max_epochs'], 
                                   atype = params['atype'], 
                                   n = 5)

eval_tab 
# epoch after which validation loss did not improve after 50 epochs, 
# loss and metrics at the end of each model iteration 
#    epochs  iter train_loss  train_f1  train_acc  val_loss    val_f1   val_acc
# 0     100     0   0.581640  0.432633   0.771858  0.584809  0.414132  0.836552
# 1     100     1   0.578635  0.436084   0.771053  0.582815  0.415887  0.835695
# 2      30     2   0.577589  0.437337   0.771394  0.579673  0.419757  0.837066
# 3      31     3   0.578685  0.435540   0.772029  0.579906  0.418761  0.838665
# 4       5     4   0.578109  0.436299   0.771687  0.581162  0.417312  0.837123
# variation in epochs ran before early stopping, however loss and metrics were consistent between runs

avg_eval # average epochs run, loss and metrics
# epochs        53.200000
# train_loss     0.578931
# train_f1       0.435579
# train_acc      0.771604
# val_loss       0.581673
# val_f1         0.417170
# val_acc        0.837020
# dtype: float64
# similar loss and performance metrics as the run without early stopping, likely because loss and metrics quickly plateaued