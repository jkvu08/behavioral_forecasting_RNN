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
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# LSTM (LSTM)                  (None, 10)                1480      
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
# array([[0.36737972, 0.3883281 , 0.04880331, 0.19548889],
#        [0.15515094, 0.67816854, 0.03508063, 0.13159987],
#        [0.2841164 , 0.42461208, 0.02623351, 0.265038  ],
#        [0.11499098, 0.7439996 , 0.02707816, 0.11393125],
#        [0.05381021, 0.8689007 , 0.02362371, 0.05366542],
#        [0.03488212, 0.9076745 , 0.01872229, 0.03872102],
#        [0.02907102, 0.9203498 , 0.01645795, 0.03412125],
#        [0.0225982 , 0.93458927, 0.01383973, 0.02897269],
#        [0.02283373, 0.93401384, 0.01400123, 0.02915114],
#        [0.02290114, 0.93384266, 0.01405505, 0.02920114]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 0, 1, 1, 1, 1, 1, 1] can see that 4 predictions differ from the target

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
# early stopping initiated at 65 epochs

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
# array([[0.43579295, 0.42106354, 0.02497299, 0.11817045],
#        [0.1270146 , 0.74329984, 0.02495599, 0.10472956],
#        [0.22866403, 0.47777608, 0.0596821 , 0.2338778 ],
#        [0.10285215, 0.7665727 , 0.03482664, 0.09574847],
#        [0.05721625, 0.86229974, 0.02631802, 0.05416593],
#        [0.03935101, 0.90645665, 0.01828163, 0.03591076],
#        [0.03625512, 0.9138584 , 0.01724655, 0.03263992],
#        [0.02793237, 0.934318  , 0.01387349, 0.02387612],
#        [0.02813731, 0.93383145, 0.01394826, 0.02408295],
#        [0.02813703, 0.9338433 , 0.01393993, 0.02407972]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 0, 3, 1, 1, 3, 1, 1, 1] can see that 5 predictions differ from the target, worse than lstm model based on first 10 predictions 

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
gru_score, _, _, _ = bmf.result_summary(y_test, 
                                         y_prob, 
                                         path, 
                                         'manual_ende_rnn_gru_evaluation') 
# gru_score = 0.322, similar performance to lstm model

# build model using f1_loss function
model = bmf.build_ende(features,
                       targets,
                       lookback,
                       n_output,
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
# early stopping activated after 74 epochs
# loss and performance curves are less noisy

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# f1_loss: 0.671
# f1: 0.330
# accuracy: 0.832

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob = y_prob.reshape(y_prob.shape[0],y_prob.shape[2]) # get rid of dummy 2nd dimension 
y_prob[0:10,:] # print subset
# array([[9.99998093e-01, 1.95273378e-06, 8.47613588e-15, 9.55769242e-09],
#        [2.40825892e-12, 1.00000000e+00, 6.98287761e-10, 1.09779864e-16],
#        [2.78764419e-05, 1.56039800e-06, 3.20231084e-05, 9.99938488e-01],
#        [2.79761492e-10, 9.99999881e-01, 1.13153305e-07, 5.50231356e-13],
#        [1.89856355e-12, 1.00000000e+00, 5.70248848e-10, 7.70473279e-17],
#        [1.52737599e-12, 1.00000000e+00, 3.86270849e-10, 4.85475610e-17],
#        [1.48254803e-12, 1.00000000e+00, 3.56785212e-10, 4.46283263e-17],
#        [1.45701821e-12, 1.00000000e+00, 3.41605938e-10, 4.25677566e-17],
#        [1.45495204e-12, 1.00000000e+00, 3.40704714e-10, 4.24331908e-17],
#        [1.45257298e-12, 1.00000000e+00, 3.39360900e-10, 4.22497056e-17]],
#       dtype=float32)

# ensure row probabilities equal to 1, might slightly deviate due to approximation of f1_loss function
# subtract a small amount from the largest class probability per row
y_proba = bmf.prob_adjust(y_prob)
y_proba[0:10,:]
# array([[9.99997973e-01, 1.95273378e-06, 8.47613588e-15, 9.55769242e-09],
#        [2.40825892e-12, 9.99999881e-01, 6.98287761e-10, 1.09779864e-16],
#        [2.78764419e-05, 1.56039800e-06, 3.20231084e-05, 9.99938369e-01],
#        [2.79761492e-10, 9.99999762e-01, 1.13153305e-07, 5.50231356e-13],
#        [1.89856355e-12, 9.99999881e-01, 5.70248848e-10, 7.70473279e-17],
#        [1.52737599e-12, 9.99999881e-01, 3.86270849e-10, 4.85475610e-17],
#        [1.48254803e-12, 9.99999881e-01, 3.56785212e-10, 4.46283263e-17],
#        [1.45701821e-12, 9.99999881e-01, 3.41605938e-10, 4.25677566e-17],
#        [1.45495204e-12, 9.99999881e-01, 3.40704714e-10, 4.24331908e-17],
#        [1.45257298e-12, 9.99999881e-01, 3.39360900e-10, 4.22497056e-17]],
#       dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_proba, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 1, 1, 1, 1, 1, 1, 1] 3 predictions differ from target, more correct compared to previous models based on first 10 records
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


model = bmf.build_ende(features,
                       targets,
                       lookback,
                       n_output, 
                       layers = params['hidden_layers'], 
                       neurons_n = params['neurons_n'], 
                       hidden_n=[params['hidden_n']],
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
# dropout_6 (Dropout)          (None, 20)                0         
# _________________________________________________________________
# dense_7 (Dense)             (None, 10)                210       
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
# 0     100  0.598755  0.416703  0.756550  0.574507  0.424599      0.836952
# 1     100  0.593344  0.421643  0.761286  0.578753  0.420451      0.839751
# 2     100  0.584978  0.429840  0.762214  0.578246  0.421083      0.840037
# 3     100  0.585035  0.429986  0.763288  0.579770  0.419283      0.836609
# 4     100  0.584877  0.429461  0.763044  0.579820  0.419510      0.834952
# similar performance across iterations, may not need to test with multiple iterations

avg_eval # average epochs run, loss and metrics
# epochs          100.000000
# loss              0.589398
# f1                0.425527
# accuracy          0.761276
# val_loss          0.578219
# val_f1            0.420985
# val_accuracy      0.837660
# # dtype: float64

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
# 0      49  0.582074  0.432602  0.760383  0.576424  0.422503      0.832724
# 1     100  0.574322  0.439943  0.770442  0.578061  0.421115      0.838723
# 2      15  0.580187  0.434362  0.765828  0.574082  0.425293      0.835295
# 3       2  0.580271  0.434497  0.769905  0.573786  0.425296      0.836723
# 4      41  0.578504  0.436215  0.767488  0.574045  0.425328      0.836209
# variation in epochs run, however loss and metrics were consistent between runs

avg_eval # average epochs run, loss and metrics
# epochs          41.400000
# loss             0.579072
# f1               0.435524
# accuracy         0.766809
# val_loss         0.575280
# val_f1           0.423907
# val_accuracy     0.835935
# dtype: float64
# similar metrics as the run without early stopping, likely because loss and metrics plateaued