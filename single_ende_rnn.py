# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021
Encoder-Decoder LSTM Model With Multivariate Input

@author: Jannet
"""
# Load libraries
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

# import datafile
dataset =  read_csv('data.csv', header = 0, index_col = 0) # VV only

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

def get_sample_weights(train_y, weights):
    """
    Generate sample weights for the for training data, since tensorflow class weights does not support 3D

    Parameters
    ----------
    train_y : training targets 
    weights : weight dictionary 
    Returns
    -------
    train_lab : array of sample weights for each label at each timestep

    """
    train_lab = to_label(train_y) # get the one-hot decoded labels for the training targets
    train_lab = train_lab.astype('float64') # convert the datatype to match the weights datatype
    for key, value in weights.items(): # replace each label with its pertaining weight according to the weights dictionary
        train_lab[train_lab == key] = value
    train_lab[train_lab == 0] = 0.0000000000001 # set zero weights as extremely low number
    return train_lab # returm the formatted sample weights

# Model Evaluation     
def confusion_mat(y, y_pred):
    """
    generates and visualizes the confusion matrix

    Parameters
    ----------
    y : labeled true values
    y_pred : labeled predictions

    Returns
    -------
    cm : confusion matrix

    """
    cm = confusion_matrix(y,y_pred, normalize = 'true') # generalize the normalized confusion matrix
    cm_figure = ConfusionMatrixDisplay(cm, display_labels = ['feeding','resting','socializing','traveling']) # visualize the confusion matrix
    cm_figure.plot() # plot results
    cm = confusion_matrix(y,y_pred) # get confusion matrix without normalization
    cm_figure = ConfusionMatrixDisplay(cm, display_labels = ['feeding','resting','socializing','traveling']) # visualize the confusion matrix
    cm_figure.plot() # plot the results 
    return cm 

def class_report(y, y_pred):
    """
    generate the class report to get the recall, precision, f1 and accuracy per class and overall

    Parameters
    ----------
    y : labeled true values
    y_pred : labeled predictions

    Returns
    -------
    class_rep : classification report

    """
 #   class_rep = classification_report(y,y_pred, zero_division = 0) # generate classification report
  #  print(class_rep) # output report 
    class_rep = classification_report(y,y_pred, zero_division = 0, output_dict = True) # generatate classification report as dictionary
    class_rep = DataFrame(class_rep).transpose() # convert dictionary to dataframe
    return class_rep 

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

# Custom metrics through callbacks (on epochs)
class F1Metrics(Callback):   
    """
    Generate epoch level F1 metrics without early stopping
    """
    # define the initial values (input into the callback function)
    def __init__(self, validation_data, train_data, verbose = 0):   
        super(F1Metrics, self).__init__()
        self.validation_data = validation_data # set validation data
        self.train_data = train_data # set the training data
        self.verbose = verbose
        
    # define values for at the beginning of training
    def on_train_begin(self, logs={}):
        self.val_f1s = [] # empty list for f1 validation
        self.train_f1s = [] # empty list for training validation
        
    # define function to be executed at the end of the epoch
    def on_epoch_end(self, validation_data, epoch, logs={}):  
        # assign, generate predictions and labels for validation dataset
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))) # generate prediction probabilities
        val_predict = to_label(val_predict) # convert probabilities to predictions
        val_targ = np.asarray(self.validation_data[1]) # assign the validation target data
        val_targ = to_label(val_targ) # convert to the labels
        
        # assign, generate predictions and labels for training dataset
        train_predict = (np.asarray(self.model.predict(self.train_data[0])))# generate prediction probabilities
        train_predict = to_label(train_predict) # convert probabilities to predictions
        train_targ = np.asarray(self.train_data[1]) # assign the validation target data
        train_targ = to_label(train_targ) # convert to the labels
        
        # if more than one timestep is predicted the predictions from each timestep need to be concatenate into a single vector
        if len(val_targ.shape) > 1: 
            val_predict = np.concatenate(val_predict) # concatenate all predictions into a vector
            val_targ = np.concatenate(val_targ) # concatenate all predictions into a vector
            train_predict = np.concatenate(train_predict) # concatenate all predictions into a vector
            train_targ = np.concatenate(train_targ) # concatenate all predictions into a vector
        
        # calcualte the f1 score
        _val_f1 = f1_score(val_targ, val_predict, average = 'macro') # for validation
        _train_f1 = f1_score(train_targ, train_predict, average = 'macro') # for training
        self.val_f1s.append(_val_f1) # append values
        self.train_f1s.append(_train_f1) # append values
        if self.verbose > 0:
            print(f' — cval_f1: {_val_f1} — ctrain_f1: {_train_f1}') # print results
        return
      
class F1EarlyStopping(Callback):  
    """
    Generate epoch level F1 metrics with early stopping
    """
    # define the initial values (input into the callback function)
    def __init__(self, validation_data, train_data, patience=50, verbose=0):   
        super(F1EarlyStopping, self).__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        self.patience = patience # threshold for early stopping
        self.verbose = verbose
        
    # define values for at the beginning of training
    def on_train_begin(self, logs={}):
        self.val_f1s = [] # empty list for the val f1
        self.train_f1s = [] # empty list for the train f1
        self.best = None # initialize the best val f1
        self.best_weights = None # initialized the best weights
        self.wait = 0 # start the wait counter at 0
        self.stopped_epoch = 0 # start the stopped epochs at 0
        self.restore_epoch = 0 # start the restore epochs at 0
        
    # define function to be executed at the end of the epoch
    def on_epoch_end(self, validation_data, epoch, logs={}):
        # assign, generate predictions and labels for validation dataset
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))) # generate prediction probabilities
        val_predict = to_label(val_predict) # convert probabilities to predictions
        val_targ = np.asarray(self.validation_data[1]) # assign the validation target data
        val_targ = to_label(val_targ) # convert to the labels
        
        # assign, generate predictions and labels for training dataset
        train_predict = (np.asarray(self.model.predict(self.train_data[0])))# generate prediction probabilities
        train_predict = to_label(train_predict) # convert probabilities to predictions
        train_targ = np.asarray(self.train_data[1]) # assign the validation target data
        train_targ = to_label(train_targ) # convert to the labels
        
        # if more than one timestep is predicted the predictions from each timestep need to be concatenate into a single vector
        if len(val_targ.shape) > 1: 
            val_predict = np.concatenate(val_predict) # concatenate all predictions into a vector
            val_targ = np.concatenate(val_targ) # concatenate all predictions into a vector
            train_predict = np.concatenate(train_predict) # concatenate all predictions into a vector
            train_targ = np.concatenate(train_targ) # concatenate all predictions into a vector
        
        # calcualte the f1 score
        _val_f1 = f1_score(val_targ, val_predict, average = 'macro') # for validation
        _train_f1 = f1_score(train_targ, train_predict, average = 'macro') # for training
        self.val_f1s.append(_val_f1) # append values
        self.train_f1s.append(_train_f1) # append values
        if self.verbose > 0:
            print(f' — cval_f1: {_val_f1} — ctrain_f1: {_train_f1}') # print results
        
        # keep track of best weights and track model improvements
        if self.best_weights is None: # if the best_weights haven't been assigned yet
            self.best_weights = self.model.get_weights() # set the first weights to be the best weight
            self.best = _val_f1 # set the first val f1 value to be the best value
            self.restore_epoch = len(self.val_f1s) # set the restored epoch to the first epoch
        else: # otherwise
            if _val_f1 >= self.best: # if current val f1 is better than or equal to the best f1
                self.best = _val_f1 # reassign the best f1 to the current one
                self.best_weights = self.model.get_weights() # reassign the best weights to the current one
                self.wait = 0 # reset the wait timer
                self.restore_epoch = len(self.val_f1s) # reassign the resotre epoch to the current one
            else: # otherwise
                self.wait +=1 # add one to the waiting counter
                if self.wait >= self.patience: # if the wait counter exceeds or is equal to the patience threshold
                    self.stopped_epoch = len(self.val_f1s) # note the current epoch for stopping
                    self.model.stop_training = True # stop the model training
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights) # reset to the best_weights

    # define function to be executed at the end of training
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0: # if early stopping was activated 
          print('Epoch %05d: early stopping' % (self.stopped_epoch)) # then print our when it was stopped
        return

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

# set the input and output length (input should not exceed 23)
n_input = 5
n_output = 1

train, test = split_dataset(dataset, 2015) # split data 
train_X, train_y, train_dft = to_supervised(train, train['TID'],1, n_input, n_output) # format training data
test_X, test_y, test_dft = to_supervised(test, test['TID'],n_output, n_input, n_output) # format testing data

# make sure data is 3D
if n_output == 1:
    test_y = test_y[:,newaxis,:]
    train_y = train_y[:,newaxis,:]
    
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