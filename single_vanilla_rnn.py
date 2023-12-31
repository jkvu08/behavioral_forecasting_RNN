# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet

Run single models of vanilla RNN using GRU and LSTM architecture and evaluate the results. This code is implemented within the training phase of the project.
This code can be used to manually tune parameters by monitoring the loss plots, confusion matrices and classification reports. 

"""
# Load libraries
import os
from pandas import read_csv
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
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
# subset and reorder predictors
datasub = datasub[['ID', 'TID', 'track_position', 'track_length', 'focal', 'year', # identifiers
                   'feed', 'rest', 'social', 'travel', # behaviors         
                   'individual_continuity', 'length', 'position', # sampling features 
                   'gestation', 'lactation', 'mating', 'nonreproductive', 'sex', # internal features 
                   'since_rest', 'since_feed', 'since_travel', # internal features
                   'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count', # external features
                   'years', 'minutes_sin', 'minutes_cos', 'doy_sin','doy_cos', # external/time features
                   'adults', 'infants', 'juveniles']] # external/group features

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
model.add(LSTM(units = neurons_n, 
               input_shape = (lookback,features), 
               name = 'LSTM')) 
# add dropout
model.add(Dropout(rate= d_rate)) 
# add dense layer & set activation function
model.add(Dense(units = hidden_n, 
                activation = 'relu', 
                kernel_initializer = 'he_uniform')) 
# add dropout
model.add(Dropout(rate= d_rate)) 
# add output layer
model.add(Dense(units = targets, 
                activation = "softmax", 
                name = 'Output')) 
model.compile(loss = 'categorical_crossentropy', # compile model, 
              optimizer = Adam(learning_rate = lr_rate), # set learning rate
              metrics= [bmf.f1,'accuracy']) # calculate metrics

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
# dense (Dense)                (None, 5)                 55        
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 5)                 0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 24        
# =================================================================
# Total params: 1,639
# Trainable params: 1,639
# Non-trainable params: 0
# _________________________________________________________________

# fit model
history = model.fit(train_X, # features
                    train_y, # targets
                    validation_data = (test_X, test_y), # add validation data
                    epochs = 50, # epochs 
                    batch_size = 512, # batch size
                    class_weight = weights, # add class weights
                    shuffle=False, # determine whether to shuffle order of data, False since we want to preserve time series
                    verbose = 2) # status print outs
history.history.keys() # examine outputs
# dict_keys(['loss', 'f1', 'accuracy', 'val_loss', 'val_f1', 'val_accuracy'])

# monitor and evaluate the results
mon_plots = bmf.monitoring_plots(history, ['loss','f1','accuracy']) # generate loss and performance curves
mon_plots.savefig(path+'manual_vanilla_rnn_lstm_monitoring.jpg', dpi=150) # save monitoring plot and examine in output file
# loss, accuracy and f1 validation all plataeu, might want to add early stopping

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.455
# f1: 0.230
# accuracy: 0.854

y_val = bmf.one_hot_decode(test_y) # retrieve labels for test targets
y_val[0:10] # view subset target labels
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob[0:10,:] # print subset
# array([[0.3491326 , 0.42578176, 0.0367919 , 0.1882938 ],
#        [0.1361342 , 0.70325327, 0.05572441, 0.10488813],
#        [0.20734158, 0.5313169 , 0.06777523, 0.19356632],
#        [0.103265  , 0.72524405, 0.06943203, 0.10205898],
#        [0.08069982, 0.80077994, 0.0506965 , 0.0678238 ],
#        [0.0650246 , 0.84106946, 0.0395254 , 0.05438046],
#        [0.05655146, 0.8612111 , 0.03411567, 0.04812174],
#        [0.0419166 , 0.89621395, 0.02489214, 0.0369774 ],
#        [0.04265027, 0.8944517 , 0.02535054, 0.03754748],
#        [0.04275464, 0.8942011 , 0.02541578, 0.03762846]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_pred[0:10] # view subset of predictions
# [1, 3, 3, 1, 1, 1, 1, 1, 1, 1] can see that 1 predictions differ from the targets of first 10 predictions

cm_fig = bmf.confusion_mat(y_val, y_pred) # generate confusion matrix 
bmf.class_report(y_val, y_pred) # generate classification report

# calculte overall f1 score and timestep f1 scores, as well as output confusion matrix and classification report in pdf
lstm_score, _, _, _ = bmf.result_summary(test_y, 
                                         y_prob, 
                                         path, 
                                         'manual_vanilla_rnn_lstm_evaluation') 
# lstm_score = 0.315

# view output file results

# build model using wrapper
model = bmf.build_rnn(features = features, 
                      targets = targets,
                      lookback = lookback,
                      neurons_n = 20, 
                      hidden_n = [10], 
                      lr_rate = 0.001, 
                      d_rate=0.3,
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
# dense_1 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,254
# Trainable params: 3,254
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
                    callbacks = [early_stopping],
                    shuffle=False,
                    verbose = 2)

# early stopping activated, stopped after 61 epochs
# monitor the results
mon_plots2 = bmf.monitoring_plots(history, ['loss','f1','accuracy'])
mon_plots2.savefig(path+'manual_vanilla_rnn_gru_monitoring.jpg', dpi=150) # save monitoring plot
# loss, acc and f1 plataeued
# could try running with f1 loss function instead

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# loss: 0.444
# f1: 0.230
# accuracy: 0.857
y_prob = model.predict(test_X) # get prediction prob for each class
y_prob[0:10,:] # print subset
# array([[0.3955299 , 0.4194195 , 0.01643234, 0.16861826],
#        [0.15616342, 0.6759525 , 0.03375596, 0.13412818],
#        [0.22671065, 0.49575683, 0.04451789, 0.23301466],
#        [0.13350278, 0.7188148 , 0.02788795, 0.11979448],
#        [0.06268971, 0.85706055, 0.01805665, 0.06219308],
#        [0.03558677, 0.91273177, 0.01334207, 0.03833944],
#        [0.02962191, 0.9248747 , 0.01268264, 0.03282075],
#        [0.02152022, 0.94540733, 0.00946264, 0.02360985],
#        [0.02209781, 0.9437576 , 0.00982062, 0.02432397],
#        [0.02222363, 0.9434247 , 0.00989556, 0.02445604]], dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_prob, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 1, 1, 1, 1, 1, 1, 1, 1] 2 predictions differ from target, one more incorrect than prior lstm model based on first 10 records
gru_score, _, _, _ = bmf.result_summary(test_y, 
                                        y_prob, 
                                        path, 
                                        'manual_vanilla_rnn_gru_evaluation')
# view output file results
# gru_score = 0.326 slightly higher f1 score compared to lstm model

# build model using f1_loss function
model = bmf.build_rnn(features = features,
                      targets = targets,
                      lookback = lookback,
                      neurons_n = 20, 
                      hidden_n = [10], 
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
# Masking (Masking)            (None, 5, 28)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                3000      
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 20)                0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_5 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,254
# Trainable params: 3,254
# Non-trainable params: 0
# _________________________________________________________________

# generate a callback for early stopping to prevent overfitting on training data
early_stopping = EarlyStopping(monitor='val_loss', # monitor validation loss
                               patience = 25, # stop if loss doesn't improve after 25 epochs
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
                    callbacks = [early_stopping],
                    shuffle=False,
                    verbose = 2)

# monitor the results
mon_plots3 = bmf.monitoring_plots(history, ['loss','f1','accuracy'])
mon_plots3.savefig(path+'manual_vanilla_rnn_gru_monitoring_f1_loss.jpg', dpi=150) # save monitoring plot
# loss and performance curves are less noisy
# early stopping not activated
# loss and metrics look like they are plateauing, 100 epochs may be sufficient

loss, f1, accuracy = model.evaluate(test_X, test_y) # evaluate the model (also could just extract from the fit directly)
# f1_loss: 0.670
# f1: 0.330
# accuracy: 0.835

y_prob = model.predict(test_X) # get prediction prob for each class
y_prob[0:10,:] # print subset
# array([[1.0000000e+00, 1.3472630e-18, 4.6596304e-16, 2.7960492e-12],
#        [9.1221800e-06, 9.9986398e-01, 4.9525370e-05, 7.7348275e-05],
#        [1.2445593e-13, 4.7163756e-15, 4.1444780e-13, 1.0000000e+00],
#        [1.3895581e-06, 4.7528458e-05, 6.6496627e-06, 9.9994445e-01],
#        [1.1786404e-07, 9.9999845e-01, 8.5626039e-07, 6.2313461e-07],
#        [4.6003908e-08, 9.9999940e-01, 3.4834852e-07, 2.4460576e-07],
#        [4.0501089e-08, 9.9999940e-01, 3.0644526e-07, 2.1201794e-07],
#        [3.6594781e-08, 9.9999940e-01, 2.8225136e-07, 1.9905660e-07],
#        [3.2992222e-08, 9.9999952e-01, 2.5652025e-07, 1.8118195e-07],
#        [3.0625234e-08, 9.9999964e-01, 2.3951239e-07, 1.6936951e-07]],
#       dtype=float32)

# ensure row probabilities equal to 1, might slightly deviate due to approximation of f1_loss function
# subtract a small amount from the largest class probability per row
y_proba = bmf.prob_adjust(y_prob)
y_proba[0:10,:]
# array([[9.9999988e-01, 1.3472630e-18, 4.6596304e-16, 2.7960492e-12],
#        [9.1221800e-06, 9.9986386e-01, 4.9525370e-05, 7.7348275e-05],
#        [1.2445593e-13, 4.7163756e-15, 4.1444780e-13, 9.9999988e-01],
#        [1.3895581e-06, 4.7528458e-05, 6.6496627e-06, 9.9994433e-01],
#        [1.1786404e-07, 9.9999833e-01, 8.5626039e-07, 6.2313461e-07],
#        [4.6003908e-08, 9.9999928e-01, 3.4834852e-07, 2.4460576e-07],
#        [4.0501089e-08, 9.9999928e-01, 3.0644526e-07, 2.1201794e-07],
#        [3.6594781e-08, 9.9999928e-01, 2.8225136e-07, 1.9905660e-07],
#        [3.2992222e-08, 9.9999940e-01, 2.5652025e-07, 1.8118195e-07],
#        [3.0625234e-08, 9.9999952e-01, 2.3951239e-07, 1.6936951e-07]],
#       dtype=float32)

# generate prediction labels
y_pred = bmf.to_label(y_proba, prob = True) # prob = True to draw from probability distribution, prob = False to pred based on max probability
y_val[0:10] # view observed targets
# [1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
y_pred[0:10] # view subset of predictions
# [0, 1, 3, 3, 1, 1, 1, 1, 1, 1] 4 predictions differ from target, worse performance from previous models based on accuracy of first 10 records
gru_score_f1, _, _, _ = bmf.result_summary(test_y, 
                                           y_prob, 
                                           path, 
                                           'manual_vanilla_rnn_gru_evaluation_f1')
# view output file results
# gru_score_f1 = 0.440, higher f1 score compared to lstm and gru model trained using categorical cross entropy loss function
# also outperforms models trained using categorical cross entropy loss function based on overall and class specific precision, recall and accuracy

# test same model multiple times
params = {'atype': 'VRNN',
          'mtype': 'GRU',
          'lookback': lookback,
          'hidden_layers': 1,
          'neurons_n': 20,
          'hidden_n': 10,
          'learning_rate': 0.001,
          'dropout_rate': 0.3,               
          'loss': False,
          'epochs': 100,
          'max_epochs': 100,
          'batch_size': 512,
          'weights_0': 1,
          'weights_1': 1,
          'weights_2': 3,
          'weights_3': 1}

model = bmf.build_rnn(features = features,
                      targets = targets,
                      lookback = params['lookback'],
                      layers = params['hidden_layers'], 
                      neurons_n = params['neurons_n'], 
                      hidden_n = [params['hidden_n']], 
                      lr_rate = params['learning_rate'], 
                      d_rate= params['dropout_rate'],
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
# dense_3 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_7 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,254
# Trainable params: 3,254
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
#    epochs      loss        f1  accuracy  val_loss    val_f1  val_accuracy
# 0     100  0.589519  0.425900  0.759260  0.574998  0.423891      0.832038
# 1     100  0.583178  0.431669  0.765950  0.578471  0.419822      0.834952
# 2     100  0.581183  0.433574  0.766414  0.576195  0.422591      0.833924
# 3     100  0.576348  0.438019  0.769099  0.574497  0.424145      0.835809
# 4     100  0.573922  0.441065  0.771565  0.574919  0.423797      0.838951
# similar performance across iterations, may not need to run multiple iterations

avg_eval # average epochs run, loss and metrics
# epochs          100.000000
# loss              0.580830
# f1                0.434045
# accuracy          0.766458
# val_loss          0.575816
# val_f1            0.422849
# val_accuracy      0.835135
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
#    epochs      loss        f1  accuracy  val_loss    val_f1  val_accuracy
# 0      38  0.572561  0.442199  0.768342  0.572085  0.426930      0.837409
# 1       1  0.572178  0.442442  0.770564  0.573464  0.426788      0.837294
# 2       3  0.568788  0.445826  0.772078  0.572362  0.426318      0.838608
# 3      10  0.572230  0.441331  0.772420  0.573192  0.426671      0.838608
# 4      22  0.570959  0.443702  0.771883  0.574267  0.424840      0.838723
# variation in epochs, however loss and metrics were consistent between runs

avg_eval # average epochs run, loss and metrics
# epochs          14.800000
# loss             0.571343
# f1               0.443100
# accuracy         0.771057
# val_loss         0.573074
# val_f1           0.426309
# val_accuracy     0.838128
# dtype: float64
# similar metrics as the run without patience, likely cause loss and metrics plateaued and only after a few epochs
# seems like early stopping can be applied given the similar performances of the models despite running for various epochs