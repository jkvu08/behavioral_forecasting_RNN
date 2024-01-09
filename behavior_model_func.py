# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:23:29 2021

@author: Jannet

Functions to compile, run and evaluate the vanilla RNN and encoder-decoder RNN models to predict the 4 state behavior of wild lemurs.

"""
import time, os, random, joblib, scipy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
from pandas import DataFrame
from numpy import newaxis
from sklearn.metrics import confusion_matrix, classification_report, f1_score, average_precision_score, roc_auc_score, log_loss, accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K
from operator import itemgetter

#########################
#### Data formatting ####
#########################
# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

def split_dataset(data, years):
    """
    Withing training, further split into training and testing data based on year
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

def to_supervised(data, TID, window, lookback, n_output = 7):
    """
    Format training data for multivariate, multistep time series prediction models

    Parameters
    ----------
    data : data
    TID : track identifier
    window : sliding window size
    lookback : lookback period
    n_output : prediction timesteps. The default is 7.
    Returns
    -------
    X : features data in array format (input)
    y : target data in array format (output)
    dft: deterministic features for prediction timesteps in array formate

    """
    # convert data into numpy array
    data = np.array(data) 
    # get empty list for X (all features), y (targets), dft (deterministic features in prediction time steps)
    # will be used to populate values 
    X, y, dft = list(), list(), list()
    # set start index as 0
    in_start = 0 
	# step over the entire dataset one time step at a time
    for _ in range(len(data)):
        in_end = in_start + lookback # define the end of the input sequence
        out_end = in_end + n_output # define the end of the output sequence
		# if we have enough data to create a full sequence and the track ID is the same for the entire sequence 
        if out_end <= len(data) and len(set(TID[in_start:out_end])) == 1:
            X.append(data[in_start:in_end, :]) # append input sequence to features list
            y.append(data[in_end:out_end, 0:4]) # append output to the targets list
            dft.append(data[in_end:out_end,7:28]) # append the deterministic features for current timestep to the deterministic features list
        in_start += window # move along the window time step
    # convert lists to array
    X = np.array(X)
    y = np.array(y)
    dft = np.array(dft)
    # delete unknown target rows since we won't be predicting those
    a = np.where(np.min(y[:,:,0],axis = 1)==-1) # extract the indices with rows that have unknown targets (i.e., values == -1)
    X = np.delete(X,a[0],axis =0) # delete unknown target rows based on indices
    y = np.delete(y,a[0],axis =0) # delete unknown target rows
    dft = np.delete(dft,a[0],axis =0) # delete unknown target rows
    if y.shape[1] == 1: # if the target is a single timestep (i.e., predicting a single timestep at a time)
        y = y.reshape((y.shape[0],y.shape[2])) # then reshape the target 3D data to 2D
        dft = dft.reshape((dft.shape[0],dft.shape[2])) # also reshape the deterministic feat from 3D to 2D data 
    return X, y, dft 

def one_hot_decode(encoded_seq):
    """
    Reverse one_hot encoding
    Arguments:
        encoded_seq: array of one-hot encoded data 
	Returns:
		Series framed for supervised learning.
	"""
    if len(encoded_seq.shape) == 1: # if single prediction to decode
        return(np.argmax(encoded_seq))
    else: # otherwise there are multiple predictions to decode
        return [np.argmax(vector) for vector in encoded_seq] # returns the index with the max value

# function to adjust the prediction prob between 0 and 1, due to precision of class weights/sample weigths and f1 loss function
def prob_adjust(y_prob):
    """
    ensure probabiltiies fall between 0 and 1 by subtracting small number (0.0001) from the largest probability

    Parameters
    ----------
    y_prob : predicted prob of classes

    Returns
    -------
    y_prob : adjusted predicted prob of classes

    """
    y_max = tuple(one_hot_decode(y_prob)) # get max prob class
    y_adjust = y_prob[range(y_prob.shape[0]), y_max] - 0.0000001 # subtract small number from max prob
    y_prob[range(y_prob.shape[0]), y_max] = y_adjust # replace adjusted prob into data
    return y_prob # return adjusted prob

########################
#### Model building ####
########################
def to_label(data, prob = False):
    
    """
    Gets the index of the maximum value in each row. Can be used to transform one-hot encoded data or probabilities to labels
    Parameters
    ----------
    data : one-hot encoded data or probability data
    prob : Boolean, False = get max value as label, True = sample from prob to get label

    Returns
    -------
    y_label : label encoded data

    """
    y_label = [] # create empty list for y_labels
    if len(data.shape) == 3 & data.shape[1] == 1:
        data = np.reshape(data, (data.shape[0],data.shape[2]))
    if len(data.shape) == 2: # if it is a one timestep prediction 
        if prob == False: # and prob is false
            y_label = np.array(one_hot_decode(data)) # then one-hot decode to get the labels
        else:
            for i in range(data.shape[0]):
                y_lab = np.random.multinomial(1, data[i,:])
                y_label.append(y_lab) # append the decoded value set to the list
            y_label = np.array(y_label)
            y_label = np.array(one_hot_decode(y_label))
    else: # otherwise 
        if prob == False:    
            for i in range(data.shape[1]): # for each timestep
                y_lab = one_hot_decode(data[:,i,:]) # one-hot decode
                y_label.append(y_lab) # append the decoded value set to the list
        else:
            for i in range(data.shape[1]): # for each timestep
                y_set = []
                for j in range(data.shape[0]):    
                    y_lab = np.random.multinomial(1, data[j,i,:])  
                    y_set.append(y_lab)
                y_set = np.array(y_set)
                y_set = np.array(one_hot_decode(y_set))
                y_label.append(y_set) # append the decoded value set to the list     
        y_label = np.column_stack(y_label) # stack the sets in the list to make an array where each column contains the decoded labels for each timestep
    return y_label  # return the labels 

def get_sample_weights(train_y, weights):
    """
    Get sample weights for the for training data, since tensorflow built-in 3D class weights is not supported 

    Parameters
    ----------
    train_y : training targets 
    weights : weight dictionary 
    Returns
    -------
    train_lab : array of sample weights for each label at each timestep

    """
    train_lab = to_label(train_y) # get the one-hot decoded labels for the training targets
    #train_lab = train_lab.astype('float64') # convert the datatype to match the weights datatype
    train_labels = np.copy(train_lab)
    for key, value in weights.items(): # replace each label with its pertaining weight according to the weights dictionary
        train_labels[train_lab == key] = value
    train_labels[train_labels == 0] = 0.0000000000001 # weight cannot be 0 so reassign to very small value if it is
    return train_labels # return the formatted sample weights
    
def f1(y_true, y_pred):
    '''
    calculate f1 metric within tensorflow keras framework    

    Parameters
    ----------
    y_true : observed value 
    y_pred : predicted value 

    Returns
    -------
    f1-score
    '''
    y_pred = K.round(y_pred) # round to get prediction from probability 
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0) # calculate true positive
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0) # calculate false positive
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) # calculate false negative
    p = tp / (tp + fp + K.epsilon()) # calculate precision
    r = tp / (tp + fn + K.epsilon()) # calculate recall
    f1 = 2*p*r / (p+r+K.epsilon()) # calculate f1-score
    f1 = tf.where(tf.math.is_nan(f1), # if nan
                  tf.zeros_like(f1), # set to 0
                  f1) # else set to f1 score
    return K.mean(f1) # return the mean f1-score

def f1_loss(y_true, y_pred):
    '''
    calculate loss f1 metric within tensorflow keras framework

    Parameters
    ----------
    y_true : observed value 
    y_pred : predicted value 

    Returns
    -------
    loss f1 metric
    '''
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0) # calculate true positive
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0) # calculate false positive
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) # calculate false negative
    p = tp / (tp + fp + K.epsilon()) # calculate precision
    r = tp / (tp + fn + K.epsilon()) # calculate recall
    f1 = 2*p*r / (p+r+K.epsilon()) # calculate f1-score
    f1 = tf.where(tf.math.is_nan(f1), # is nan
                  tf.zeros_like(f1),# set to 0
                  f1) # else set to f1 score
    return 1 - K.mean(f1) # calculate loss f1

def build_rnn(features, targets, lookback, neurons_n=10, hidden_n=[10], learning_rate=0.001, dropout_rate = 0.2, layers = 1, mtype = 'LSTM', cat_loss = True):
    """
    Vanilla LSTM for single timestep output prediction (one-to-one or many-to-one)
    
    Parameters
    ----------
    features: int, number of features
    targets: int, number of targets to predict
    lookback: int, number of timesteps prior to use for prediction 
    neurons_n : int,number of neurons, (the default is 10).
    hidden_n : list lenght of layers with number of hidden neurons, (the default is 10).
    learning_rate : float, learning rate (the default is 0.001).
    dropout_rate : float, drop out rate (the default is 0.2).
    layers: 1 
    mtype: string, model type (LSTM or GRU only, default is LSTM)
    cat_loss: boolean, loss type of True categorical crossentrophy loss is used, if False f1 loss is used. The default is True, 
    
    Returns
    -------
    model : model
    """

    # create an empty sequential shell 
    model = Sequential() 
    # add a masking layer to tell the model to ignore missing values (i.e., values of -1)
    model.add(Masking(mask_value = -1, 
                      input_shape = (lookback, features), 
                      name = 'Masking')) 
    # set RNN type
    if mtype == 'LSTM': # if the model type is LSTM
        # add LSTM layer
        model.add(LSTM(units =neurons_n, 
                       input_shape = (lookback,features), 
                       name = 'LSTM')) 
    else:
        # add GRU layer
        model.add(GRU(units = neurons_n, 
                      input_shape = (lookback,features), 
                      name = 'GRU')) # set the RNN type
    # add drop out
    model.add(Dropout(rate= dropout_rate)) 
    for i in range(layers): # for each hidden layer
        # add dense layer
        model.add(Dense(units = hidden_n[i], 
                        activation = 'relu', 
                        kernel_initializer =  'he_uniform')) 
        model.add(Dropout(rate = dropout_rate)) # add dropout
    # add output layer
    model.add(Dense(units = targets, 
                    activation = "softmax", 
                    name = 'Output')) 
    # compile model 
    if cat_loss == True: # if true
        model.compile(loss = 'categorical_crossentropy', # use categorical crossentropy loss
                      optimizer = Adam(learning_rate = learning_rate), # set learning rate 
                      metrics = [f1, 'accuracy']) # monitor metrics
    else: 
        model.compile(loss = f1_loss, # otherwise use f1 loss 
                      optimizer = Adam(learning_rate = learning_rate), # set learning rate 
                      metrics = [f1, 'accuracy']) # monitor metrics
    return model 

def build_ende(features, targets, lookback, n_outputs, neurons_n0 = 10, neurons_n1 = 10, hidden_n = [10], td_neurons = 10, learning_rate  = 0.001, dropout_rate = 0.2, layers = 1, mtype = 'LSTM', cat_loss = True):
    """
    Single encoder-decoder model

    Parameters
    ----------
    features: int, number of features
    targets: int, number of targets to predict
    n_outputs: int, number of timesteps to predict
    lookback: int, number of timesteps prior to use for prediction
    neurons_n0 : number of neurons for first RNN layer. The default is 10.
    neurons_n1 : number of neurons for second RNN layer. The default is 10.  
    hidden_n : list of number of hidden neurons. The default is 10.
    td_neurons : number of timde distributed neurons. The default is 10.
    learning_rate : Learning rate. The default is 0.001.
    dropout_rate : dropout rate. The default is 0.2.
    layers : number of layers The default is 1.
    mtype : model type, should be LSTM or GRU. The default is 'LSTM'.
    cat_loss: boolean, default is True, categorical cross-entrophy loss is used, if False f1 loss is used

    Returns
    -------
    model : compiled model

    """   
    # create an empty sequential shell 
    model = Sequential() 
    # add a masking layer to tell the model to ignore missing values (i.e., values of -1)
    model.add(Masking(mask_value = -1, 
                      input_shape = (lookback, features), 
                      name = 'Masking')) 
    # set RNN type
    if mtype == 'LSTM': # if the model is an LSTM
        # add LSTM layer
        model.add(LSTM(units =neurons_n0, 
                       input_shape = (lookback,features), 
                       name = 'LSTM')) 
    else: # otherwise set the GRU as the model type
        # add GRU layer
        model.add(GRU(units = neurons_n0, 
                      input_shape = (lookback,features), 
                      name = 'GRU'))
    # add a dropout layer
    model.add(Dropout(rate = dropout_rate))
    for i in range(layers): # for each hidden layer  
        # add a dense layer
        model.add(Dense(units = hidden_n[i], 
                        activation = 'relu', 
                        kernel_initializer =  'he_uniform')) 
        # add a dropout layer
        model.add(Dropout(rate = dropout_rate))
    model.add(RepeatVector(n_outputs)) # repeats encoder context for each prediction timestep
    # add approriate RNN type after repeat vector
    if mtype == 'LSTM': 
        model.add(LSTM(units = neurons_n1, 
                       input_shape = (lookback,features), 
                       return_sequences=True)) 
    else: # else set the layer to GRU
        model.add(GRU(units = neurons_n1, 
                      input_shape = (lookback,features),
                      return_sequences = True)) 
    # make sequential predictions, applies decoder fully connected layer to each prediction timestep
    model.add(TimeDistributed(Dense(units = td_neurons, activation='relu')))
    # applies output layer to each prediction timestep
    model.add(TimeDistributed(Dense(targets, activation = "softmax"))) 
    # compile model 
    if cat_loss == True: # if true
        # compile model 
        model.compile(loss = 'categorical_crossentropy', # use categorical crossentropy loss
                      optimizer = Adam(learning_rate = learning_rate), # set learning rate 
                      metrics = [f1,'accuracy'], # monitor metrics
                      sample_weight_mode = 'temporal') # add sample weights, since class weights are not supported in 3D
    else: 
        model.compile(loss = f1_loss, # otherwise use f1 loss 
                      optimizer = Adam(learning_rate = learning_rate), # set learning rate 
                      metrics = [f1,'accuracy'], # monitor metrics
                      sample_weight_mode = 'temporal') # add sample weights, since class weights are not supported in 3D
    return model

def build_model_func(params, features, targets):
    '''
    construct vanilla RNN or encoder-decode RNN based on parameter dictionary specifications

    Parameters
    ----------
    params : dict, dictionary of paramters and hyperparameters
    features : int, number of features used for prediction
    targets : int, number of targets (classes) predicted

    Raises
    ------
    Exception
        something other than 'VRNN' designating vanilla RNN or 'ENDE' designating encoder-decoder RNN was specified in params['atype']

    Returns
    -------
    model : RNN model

    '''
    if params['atype'] == 'VRNN':
        model = build_rnn(features, 
                              targets, 
                              lookback = params['lookback'], 
                              neurons_n = params['neurons_n'],
                              hidden_n = [params['hidden_n0'],params['hidden_n1']],
                              learning_rate =params['learning_rate'],
                              dropout_rate = params['dropout_rate'],
                              layers = params['hidden_layers'], 
                              mtype = params['mtype'], 
                              cat_loss = params['loss'])
    elif params['atype'] == 'ENDE':
        model = build_ende(features, 
                               targets, 
                               lookback = params['lookback'], 
                               n_outputs = params['n_outputs'], 
                               neurons_n0 = params['neurons_n0'],
                               neurons_n1 = params['neurons_n1'],
                               hidden_n = [params['hidden_n0'],params['hidden_n1']],
                               td_neurons = params['td_neurons'], 
                               learning_rate =params['learning_rate'],
                               dropout_rate = params['dropout_rate'],
                               layers = params['hidden_layers'], 
                               mtype = params['mtype'],
                               cat_loss = params['loss'])
    else:
        raise Exception ('invalid model architecture')    
    return model

##########################
#### Model evaluation ####
##########################
def monitoring_plots(result, metrics):
    """
    plot the training and validation loss, f1 and accuracy

    Parameters
    ----------
    result : history from the fitted model 

    Returns
    -------
    monitoring plots outputted
    """
    n = len(metrics)
    fig, ax = plt.subplots(1,n, figsize = (2*n+2,2))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.plot(result.history[metrics[i]], label='train')
        plt.plot(result.history['val_'+metrics[i]], label='validation')
        plt.legend()
        plt.title(metrics[i])
    fig.tight_layout()
    return fig
    
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
    labels = ['feeding','resting','socializing','traveling']
    fig, ax = plt.subplots(1,2,figsize=(6,2))
    cm_norm = confusion_matrix(y,y_pred, normalize = 'true') # normalized confusion matrix to get proportion instead of counts
    cm_count = confusion_matrix(y,y_pred) # get confusion matrix without normalization (i.e., counts)
    sns.set(font_scale=0.5)
    plt.subplot(1,2,1)
    sns.heatmap(cm_count, 
                    xticklabels=labels, 
                    yticklabels=labels, 
                    annot=True, 
                    fmt ='d') 
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.ylabel('True label', fontsize = 7)
    plt.xlabel('Predicted label', fontsize = 7)
    plt.subplot(1,2,2)
    sns.heatmap(cm_norm, 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=True) 
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.ylabel('True label', fontsize = 7)
    plt.xlabel('Predicted label', fontsize = 7)
    fig.tight_layout()
    plt.show(block=True)
    return fig

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
    class_rep = classification_report(y,y_pred, zero_division = 0, output_dict = True) # generatate classification report as dictionary
    class_rep = DataFrame(class_rep).transpose() # convert dictionary to dataframe
    return class_rep    

def result_summary(test_y, y_prob, path, filename):
    """
    Summary of model evaluation for single model for multiple iterations. Generates the prediction timestep and overall F1 score, overall 
    classification report and confusion matrix. Outputs classification report nad confusion matrix to pdf.
    
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
    y_label = to_label(test_y) # observed target
    y_pred = to_label(y_prob, prob = True) # predicted target
    
    if len(y_label.shape) == 1: # no multiple scores
        scores = 'nan'
    else:
        scores = [] # create empty list to populate with timestep level predictions
        for i in range(len(y_pred)): # for each timestep
            f1 = f1_score(y_label[i,:], y_pred[i,:], average = 'macro') # get the f1 value at the timestep
            scores.append(f1) # append to the empty scores list
        y_pred = np.concatenate(y_pred) # merge predictions across timesteps to single vector
        y_label = np.concatenate(y_label) # merge target values across timesteps to single vector
    print('sequence level f1 score: ', scores)
    score = f1_score(y_label, y_pred, average = 'macro') # generate the overall f1 score
    print('overall f1 score: ', score)
    
    class_rep = class_report(y_label, y_pred) # get class report for overall
    
    with PdfPages(path+filename+'.pdf') as pdf:
        cm = confusion_mat(y_label, y_pred) # get confusion matrix for overall
        pdf.savefig(cm) # save figure
        plt.close() # close page
        plt.figure(figsize=(6, 2)) # assign figure size
        plt.table(cellText=np.round(class_rep.values,4),
                      colLabels = class_rep.columns, 
                      rowLabels=class_rep.index,
                      loc='center',
                      fontsize = 9)
        plt.axis('tight') 
        plt.axis('off')
        pdf.savefig() # save figure
        plt.close() # close page
    return score, scores, class_rep, cm

def eval_iter(model, params, train_X, train_y, test_X, test_y, patience = 0 , max_epochs = 300, atype = 'VRNN', n = 1):
    """
    pipeline for fitting and evaluating the model n number of times. 

    Parameters
    ----------
    model : model
    params : hyperparameters
    train_X : training features
    train_y :  training targets
    test_X : testing features
    test_y : testing targets
    patience: nonegative integer early stopping patience value. Default is 0
    max_epochs: number of epochs to run. Default is 300
    atype: architecture type (VRNN or ENDE). Default is VRNN
    n : number of iterations to run for a single model. Default is 1

    Returns
    -------
    eval_df : dataframe of epochs, loss, and metrics for each iteration
    avg_val: average of the epochs, loss, and metrics (training and validation) across iterations 
    """
    # assign class weights
    weights = dict(zip([0,1,2,3], 
                       [params['weights_0'], 
                        params['weights_1'], 
                        params['weights_2'], 
                        params['weights_3']]))
    # assign the callback and weight type based on the model type
    if atype == 'VRNN':
        class_weights = weights # assign class weights as weights
        sample_weights = None
    elif atype == 'ENDE':
        class_weights = None 
        sample_weights = get_sample_weights(train_y, weights) # generate the formatted sample weights 
    else:
        raise Exception ('invalid model type')
    eval_run = [] # create empty list for the evaluations
    if patience > 0:
        early_stopping = EarlyStopping(patience = patience, monitor='val_loss', mode = 'min', restore_best_weights=True, verbose=0)
        callback = [early_stopping]
    else:
        callback = None
    for i in range(n): # for each iteration
        # fit the model 
        history = model.fit(train_X, 
                            train_y, 
                            validation_data = (test_X, test_y),
                            epochs = max_epochs, 
                            batch_size = params['batch_size'],
                            sample_weight = sample_weights,
                            class_weight = class_weights,
                            verbose = 2,
                            shuffle=False,
                            callbacks = callback)
        mod_eval = []
        # pull out monitoring metrics
        if len(history.history['loss']) == max_epochs: # if early stopping not activated then
            mod_eval.append(max_epochs) # assign the epochs to the maximum epochs
            mod_eval.append(i) # assign iteration number 
            for v in history.history.values():
                mod_eval.append(v[-1]) # append ending metrics
        else: # otherwise if early stopping was activate
            mod_eval.append(len(history.history['loss'])-patience) # assign stopping epoch as the epoch before improvements dropped
            mod_eval.append(i) # assign iteration number 
            for v in history.history.values():
                mod_eval.append(v[-patience-1]) # append ending metrics
        eval_run.append(mod_eval)
    eval_df = DataFrame(eval_run, 
                            columns = ['epochs','iter','train_loss','train_f1','train_acc',
                                       'val_loss','val_f1','val_acc'])
    avg_val = eval_df.mean(axis =0)
    return eval_df, avg_val.loc[avg_val.index != 'iter']

################################
#### Predictive Performance ####
################################
def best_params(path, filename, metric = 'val_f1', minimize = False):
    """
    Extract best model parameters

    Parameters
    ----------
    path : str,
        file directory of hyperopt result
    filename : str,
        filename of hyperopt result 
    metric : str, optional
        metric to determine the best model. The default is 'val_f1'.
    minimize : bool, optional
        whether to minimize or maximize the metric. The default is False.

    Returns
    -------
    space_best : best model parameters
    """
    trials = joblib.load(path + filename + '.pkl')
    rnn_df = pd.read_csv(path + filename +'.csv', header = 0, index_col = 0)
    if minimize == True:
        bid = rnn_df[rnn_df[metric] == max(rnn_df[metric])].index.values[0]
    else:
        bid = rnn_df[rnn_df[metric] == min(rnn_df[metric])].index.values[0]
    space_best = trials.results[bid]['params']
    return space_best

def train_test_format(train,test, params):
    """
    format the training and testing data for the model

    Parameters
    ----------
    train : training data 
    test : testing data
    params : model parameters

    Raises
    ------
    Exception
        something other than 'full','behavior','internal','external' designated in params['predictor']
    

    Returns
    -------
    train_X : array,
        training features
    train_y : array,
        training targets (one-hot encoded)
    train_dft : array,
        deterministic training features
    test_X : array,
        testing features
    test_y : array,
        testing targets (one-hot encoded)
    test_dft : array,
        deterministic testing features

    """
    # format training data
    train_X, train_y, train_dft = to_supervised(data = train.iloc[:,6:34], 
                                                TID = train['TID'], 
                                                window = 1, 
                                                lookback = params['lookback'], 
                                                n_output = params['n_outputs']) 
    # format testing data
    test_X, test_y, test_dft = to_supervised(data = test.iloc[:,6:34], 
                                             TID = test['TID'],
                                             window = 1, 
                                             lookback = params['lookback'], 
                                             n_output = params['n_outputs'])
    
    # if encoder-decode model and predict 1 timestep, reconfigure 2d y to 3d
    if params['atype'] == 'ENDE' and params['n_outputs'] == 1:
        test_y = test_y[:,newaxis,:]
        train_y = train_y[:,newaxis,:]
    
    # assign and format feature set
    if params['predictor'] == 'full': # use full set of features
        train_X = train_X
        test_X = test_X
    elif params['predictor'] == 'behavior': # use only prior behaviors as features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:4]) 
        test_X = np.copy(test_X[:,:,0:4])
    elif params['predictor'] == 'internal': # use internal features (behaviors and sex) as features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:12]) 
        test_X = np.copy(test_X[:,:,0:12])
    elif params['predictor'] == 'external': # use the extrinsic conditions
        # subset only extrinsic features
        train_X = np.copy(train_X[:,:,8:25])
        test_X = np.copy(test_X[:,:,8:25])  
    else:
        raise Exception ('invalid feature selection')   
    
    return train_X, train_y, train_dft, test_X, test_y, test_dft

def model_assess(train, test, params):
    """
    Run and assess predictive performance of models

    Parameters
    ----------
    train : training dataset
    test : testing dataset
    params : model parameters
        
    Raises
    ------
    Exception
        something other than 'full','behavior','internal','external' designated in params['predictor']
    
    Returns
    -------
    results_dict : dictionary 
        {'model': model
        'history': fitted model results
        'confusion_matrix': confusion matrix
        'report': classification report
        'predictions': testing dataframe with predictions 
        'train_X': training features
        'train_y': training targets
        'test_X': testing features
        'test_y': testing targets (one-hot encoded)
        'y_label': labelled testing targets
        'y_pred': predictions, drawn from prob
        'y_predmax': predictions, max prob
        'y_prob': multiclass prediction probabilities
        'evals': llist of loss, precision-recall AUC and receiver-operator AUC 
        'params': parameters
        }
        
    """
    start_time = time.time()
    # format training and testing data
    train_X, train_y, train_dft, test_X, test_y, test_dft = train_test_format(train, test, params)
    
    targets = 4 # set number of targets (4 behavior classes)
    features = test_X.shape[2] # get number of features 
    
    model = build_model_func(params, features, targets) # build model
    
    # assign class weights
    weights = dict(zip([0,1,2,3], 
                       [params['weights_0'], 
                        params['weights_1'], 
                        params['weights_2'], 
                        params['weights_3']]))
    # assign the callback and weight type based on the model type
    if params['atype'] == 'VRNN':
        class_weights = weights # assign class weights as weights
        sample_weights = None
    elif params['atype'] == 'ENDE':
        class_weights = None 
        sample_weights = get_sample_weights(train_y, weights) # generate the formatted sample weights 
    else:
        raise Exception ('invalid model type')
    # fit the model 
    history = model.fit(train_X, 
                        train_y,
                        epochs = params['max_epochs'], 
                        batch_size = params['batch_size'],
                        sample_weight = sample_weights,
                        class_weight = class_weights,
                        verbose = 2,
                        shuffle=False)
     
    y_prob = model.predict(test_X)
    y_label = to_label(test_y) # observed target
    y_pred = to_label(y_prob, prob = True) # predicted target
    y_predmax = to_label(y_prob, prob = False) # predicted target
    
    cm = confusion_mat(y_label, y_pred) # get confusion matrix for overall
    class_rep = class_report(y_label, y_pred) # get class report for overall
    # calculate ROC AUC
    macro_auc = roc_auc_score(y_label, y_prob,average = 'macro',multi_class ='ovo')
    weighted_auc = roc_auc_score(y_label, y_prob,average = 'weighted',multi_class ='ovo')
    micro_auc = roc_auc_score(y_label, y_prob,multi_class ='ovr')
    # calculate PR AUC
    macro_pr = average_precision_score(test_y, y_prob) # calculate area under the precision-recall curve unweighted
    weighted_pr = average_precision_score(test_y, y_prob, average = 'weighted') # calculate area under the precision-recall curve weighted 
    loss = log_loss(y_label, y_prob) # calculate loss
    
    # concatenate into one dataframe
    results_df = np.column_stack((test_dft, y_label, y_pred, y_predmax, y_prob))
    names = ['sex', 'gestation', 'lactation', 'mating', 'nonreproductive',
       'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count',
       'years', 'minutes_sin', 'minutes_cos', 'doy_sin', 'doy_cos', 'adults',
       'infants', 'juveniles', 'individual_continuity', 'length', 'position',
       'obs','pred','predmax','feed_prob','rest_prob','social_prob','travel_prob']
       
    pred_df = DataFrame(results_df, columns = names)
    
    results_dict = {'model': model,
                    'history': history,
                    'confusion_matrix': cm, 
                    'report': class_rep, 
                    'predictions': pred_df,
                    'train_X': train_X,
                    'train_y': train_y,
                    'test_X': test_X,
                    'test_y': test_y,
                    'test_dft': test_dft,
                    'y_label': y_label,
                    'y_pred': y_pred,
                    'y_predmax': y_predmax,
                    'y_prob': y_prob,
                    'evals': [weighted_auc, micro_auc, macro_auc, weighted_pr, macro_pr, loss],
                    'params': params
                    }
    
    print('took', (time.time()-start_time)/60, 'minutes') # print the time lapsed 
    return results_dict

def daily_dist(df, prob = True):
    """
    Get the daily distribution of behaviors
    
    Parameters
    ----------
    df : dataframe,
        dataframe of behavioral data with observed and predicted behavior classes
    prob: bool,
        True when prediction draw from probilities, False when prediction taken as highest probability. Default is True

    Returns
    -------
    prop_df : dataframe,
                daily proporiton of observed and predicted behaviors.

    """
    # if isinstance(df, np.ndarray): # if the data is a numpy array, convert it and assign the column names
    #     df = DataFrame(df, columns = datasub.columns.values[(7+4):(7+18)].tolist() +['y','y_pred'])
    df['ID'] = df['years'].astype(str) + df['doy_sin'].astype(str) + df['doy_cos'].astype(str) # create date identifier
    df_freq = df.value_counts(['ID'],dropna=False).reset_index() # get counts for the records per day 
    # count each observed type of behavior class per day 
    df_y = df.groupby(['ID','obs']).obs.count()
    levels = [df_y.index.levels[0].values,list(range(4))]
    new_index = pd.MultiIndex.from_product(levels, names=['ID','behavior']) # get indices 
    df_y = df_y.reindex(new_index,fill_value=0).reset_index() # reindex
    # count each predicted type of behavior class per day 
    if prob == True:    
        df_pred = df.groupby(['ID','pred']).pred.count()
        df_pred = df_pred.reindex(new_index,fill_value=0).reset_index() # reindex
    else:
        df_pred = df.groupby(['ID','predmax']).predmax.count()
        df_pred = df_pred.reindex(new_index,fill_value=0).reset_index() # reindex
        
    # concatenate the data into one dataframe
    # merge the behavior and predicted behavior counts 
    prop_df = pd.merge(left =df_y, 
                       right = df_pred, 
                       left_on = ["ID","behavior"], 
                       right_on= ["ID","behavior"], 
                       how = 'outer') 
    
    # merge the daily behavior counts and predictions with the number of records per day
    prop_df = pd.merge(left = prop_df, 
                       right = df_freq, 
                       on = ["ID"], 
                       how = 'left') 
    prop_df.rename(columns ={0:'n'}, inplace = True) # rename the relevant columns

    # get proportions
    prop_df['y_prop'] = prop_df.obs/prop_df.n 
    if prob == True:
        prop_df['ypred_prop'] = prop_df.pred/prop_df.n
    else:    
        prop_df['ypred_prop'] = prop_df.predmax/prop_df.n
    return prop_df

def daily_dist_plot(df):
    """
    Parameters
    ----------
    df : plots with proportions of daily behaviors 
    
    Returns
    -------
    fig : figure 

    """
    # set graph layout
    behaviors = ['feed','rest','social','travel']
    fig, axs = plt.subplots(2, 4, 
                            tight_layout = True, 
                            figsize=(8,4))
    bins = np.arange(0,1.1,0.1)
    for i in range(4):
        axs[1,i].hist(df[df['obs'] == i].y_prop, 
                      range = (0,1), 
                      bins = bins, 
                      alpha = 0.5, 
                      label = 'true', 
                      density = False)
        axs[1,i].hist(df[df['pred'] == i].ypred_prop, 
                      range = (0,1),  
                      bins = bins, 
                      alpha = 0.5, 
                      label = 'predicted', 
                      density = False)
        axs[0,i].hist([df[df['obs'] == i].y_prop, 
                       df[df['pred'] == i].ypred_prop], 
                      bins = bins, 
                      label=['true', 'predicted'])
        axs[0,i].set_title(behaviors[i])
    
    fig.text(0.5, 0.00, 
             'proportion', 
             ha='center')
    fig.text(0.00, 0.5, 
             'frequency', 
             va='center', 
             rotation='vertical')
    handles, labels = axs[0,1].get_legend_handles_labels()
    fig.legend(handles, 
               labels, 
               loc = (0.87, 0.8), 
               prop={'size': 7})
    return fig
        
def perm_feat(start_index, end_index, t, eval_X):
    """

    Parameters
    ----------
    start_index : first indices of for the columns that pertain to the multi-column variables
    end_index : last indices + 1 of for the columns that pertain to the multi-column variables
    t : is lookback timestep being evaluated
    eval_X : features dataset

    Returns
    -------
    eval_X_copy : data with permutated multiclass variable

    """
    eval_X_copy = np.copy(eval_X) # make copy of the original features
    value = np.copy(eval_X_copy[:,t,start_index:end_index]) # make a copy of columns
    eval_X_copy[:,t,start_index:end_index] = np.random.permutation(value) # permute the rows and replace the values in the copied df
    return eval_X_copy 

def perm_assess(model, X_reshape): 
    """
    Parameters
    ----------
    model : fitted model
    X_reshape : feature data
    y : target data
    Returns
    -------
    dict
        confusion_matrix : confusion matrix output
        report: classification report output
    """
    # make a prediction
    y_prob = model.predict(X_reshape)
    y_pred = to_label(y_prob,prob = True)
    y_predmax = to_label(y_prob, prob = False) # predicted target
    
    return y_pred, y_prob, y_predmax

def algo_var(test_y, test_dft, y_label, y_pred, y_predmax, y_prob, name, lookback = -99, prob = True):
    """
    returns performance metrics for a model

    Parameters
    ----------
    test_y : array,
        testing targets (one-hot encoded)
    test_dft : array,
        testing deterministic features
        DESCRIPTION.
    y_label : array,
        labelled testing targets.
    y_pred : array,
        labelled class predictions drawn from probability distribution
    y_predmax: array 
        labelled class predictions, predicted based on max probability
    y_prob : array,
        multiclass prediction probabilities
    name : str,
        model name to be used as identifier.
    lookback : int,
        lookback. optional default is -99 (missing value used for non-permuted metrics)
    prob: bool,
        True when prediction draw from probilities, False when prediction taken as highest probability. Default is True
    
    Returns
    -------
    df : dataframe,
        model results

    """
    # plot confusion matrix
    cm = confusion_matrix(y_label, y_pred)
    # generate classification reports
    class_rep = class_report(y_label, 
                             y_pred)
    
    # calculate ROC AUC 
    macro_auc = roc_auc_score(y_label, y_prob,average = 'macro',multi_class ='ovo')
    weighted_auc = roc_auc_score(y_label, y_prob, average = 'weighted',multi_class ='ovo')
    micro_auc = roc_auc_score(y_label, y_prob,multi_class ='ovr')
    
    # calculate PR AUC
    pr_auc_mac = average_precision_score(test_y, y_prob) # calculate area under the precision-recall curve unweighted
    pr_auc_weight = average_precision_score(test_y, y_prob, average = 'weighted') # calculate area under the precision-recall curve weighted 
    loss = log_loss(y_label, y_prob) # calculate loss
    
    # column names
    cnames = ['sex', 'gestation', 'lactation','mating', 'nonreproductive', 'fragment', 'rain', 'temperature', 
              'flower_count', 'fruit_count','years', 'minutes_sin', 'minutes_cos', 'doy_sin', 'doy_cos', 'adults',
              'infants', 'juveniles', 'individual_continuity', 'length', 'position', 'obs','pred','predmax','feed_prob',
              'rest_prob','social_prob','travel_prob']
    
    pred_df = np.column_stack((test_dft, y_label, y_pred, y_predmax, y_prob))
    pred_df = DataFrame(pred_df, columns = cnames)
    
    # extract metrics
    datarow = [name,lookback] # 2
    overall_acc = [class_rep.iloc[4,0]]
    datarow = datarow + overall_acc # 1 
    
    overall_met = list(class_rep.iloc[5,0:3]) # overall precision, recall and f1 score (3)
    datarow = datarow + overall_met # 3
    
    acc = [accuracy_score(np.array(y_label) == i, np.array(y_pred) == i) for i in range(4)] # class accuracy scores
    datarow = datarow + acc # 4
    
    f1 = list(class_rep.iloc[0:4,2]) # class f1 scoores (4)
    datarow = datarow + f1 # 4
    
    prec = list(class_rep.iloc[0:4,0]) # class precision scores (4)
    datarow = datarow + prec # 4
    
    recall = list(class_rep.iloc[0:4,1]) # class recall scores (4)
    datarow = datarow + recall # 4
    
    
    datarow = datarow + [macro_auc, weighted_auc, micro_auc, pr_auc_mac, pr_auc_weight, loss]
    
    # mean scores without socializing class
    metrics_3 = [np.mean(itemgetter(0,1,3)(acc)), 
                 np.mean(itemgetter(0,1,3)(prec)),
                 np.mean(itemgetter(0,1,3)(recall)),
                 np.mean(itemgetter(0,1,3)(f1))] 
    datarow = datarow + metrics_3 # (4)
        
    conmat = np.reshape(cm,(1,16)) # add confusion matrix values (16)
    datarow = datarow + list(conmat.ravel()) 
    datarow = datarow + list(np.sum(cm,0))
    
    t_prop = daily_dist(pred_df, prob = prob) # get the daily proportions
    for i in [0,1,2,3]: 
        ks = scipy.stats.ks_2samp(t_prop[t_prop['behavior'] == i].y_prop, 
                                  t_prop[t_prop['behavior'] == i].ypred_prop, 
                                  alternative = 'two_sided') # get the d statistics and p-values for the KS test
        datarow = datarow + list(ks) # add KS values (6)
    
    mean_df = t_prop.groupby('behavior').mean('y_prop')
    mean_val = list(mean_df.ypred_prop.values)
    datarow = datarow + mean_val
  
    df = pd.DataFrame({'feature': pd.Series(dtype = 'str'),
                       'lookback': pd.Series(dtype = 'int'),
                       'accuracy': pd.Series(dtype = 'float'),
                       'precision': pd.Series(dtype = 'float'),
                       'recall': pd.Series(dtype = 'float'),
                       'f1': pd.Series(dtype = 'float'),
                       'accuracy_f': pd.Series(dtype = 'float'),
                       'accuracy_r': pd.Series(dtype = 'float'),
                       'accuracy_s': pd.Series(dtype = 'float'),
                       'accuracy_t': pd.Series(dtype = 'float'),
                       'f1_f': pd.Series(dtype = 'float'),
                       'f1_r': pd.Series(dtype = 'float'),
                       'f1_s': pd.Series(dtype = 'float'),
                       'f1_t': pd.Series(dtype = 'float'),
                       'precision_f': pd.Series(dtype = 'float'),
                       'precision_r': pd.Series(dtype = 'float'),
                       'precision_s': pd.Series(dtype = 'float'),
                       'precision_t': pd.Series(dtype = 'float'),
                       'recall_f': pd.Series(dtype = 'float'),
                       'recall_r': pd.Series(dtype = 'float'),
                       'recall_s': pd.Series(dtype = 'float'),
                       'recall_t': pd.Series(dtype = 'float'),
                       'roc_weight': pd.Series(dtype = 'float'), 
                       'roc_micro': pd.Series(dtype = 'float'),
                       'roc_macro': pd.Series(dtype = 'float'),
                       'pr_weight': pd.Series(dtype = 'float'),
                       'pr_macro': pd.Series(dtype = 'float'),
                       'cat_loss': pd.Series(dtype = 'float'),
                       'accuracy_3': pd.Series(dtype = 'float'),
                       'precision_3': pd.Series(dtype = 'float'),
                       'recall_3': pd.Series(dtype = 'float'),
                       'f1_3': pd.Series(dtype = 'float'),
                       'FF': pd.Series(dtype = 'int'),
                       'FR': pd.Series(dtype = 'int'),
                       'FS': pd.Series(dtype = 'int'),
                       'FT': pd.Series(dtype = 'int'),
                       'RF': pd.Series(dtype = 'int'),
                       'RR': pd.Series(dtype = 'int'),
                       'RS': pd.Series(dtype = 'int'),
                       'RT': pd.Series(dtype = 'int'),
                       'SF': pd.Series(dtype = 'int'),
                       'SR': pd.Series(dtype = 'int'),
                       'SS': pd.Series(dtype = 'int'),
                       'ST': pd.Series(dtype = 'int'),
                       'TF': pd.Series(dtype = 'int'),
                       'TR': pd.Series(dtype = 'int'),
                       'TS': pd.Series(dtype = 'int'),
                       'TT': pd.Series(dtype = 'int'),
                       'F_pred': pd.Series(dtype = 'int'),
                       'R_pred': pd.Series(dtype = 'int'),
                       'S_pred': pd.Series(dtype = 'int'),
                       'T_pred': pd.Series(dtype = 'float'),
                       'KSD_F': pd.Series(dtype = 'float'),
                       'KSP_F': pd.Series(dtype = 'float'),
                       'KSD_R': pd.Series(dtype = 'float'),
                       'KSP_R': pd.Series(dtype = 'float'),
                       'KSD_S': pd.Series(dtype = 'float'),
                       'KSP_S': pd.Series(dtype = 'float'),
                       'KSD_T': pd.Series(dtype = 'float'),
                       'KSP_T': pd.Series(dtype = 'float'),
                       'F_dprop': pd.Series(dtype = 'float'),
                       'R_dprop': pd.Series(dtype = 'float'),
                       'S_dprop': pd.Series(dtype = 'float'),
                       'T_dprop': pd.Series(dtype = 'float')})
    
    df.loc[len(datarow)] = datarow
    
    return df

def perm_behavior(model, df, test_X, test_dft, test_y, y_label, seed, name, path, prob = True):
    """
    permutate each behavior in each timestep and recalcualte performance metrics

    Parameters
    ----------
    model : tensor network,
        model
    df : dataframe,
        original performance metrics (non-permuted)
    test_X : array,
        testing features
    test_dft : array,
        testing deterministic features
    test_y : array,
        testing targets (one-hot encoded)
    y_label : array,
        testing targets, labelled 
    seed : int,
        random seed
    name : str, 
        filename to save
    path: str,
        path directory to save
    prob: bool, optional
        True when prediction draw from probilities, False when prediction taken as highest probability. The default is True
 
    Returns
    -------
    df : dataframe,
        permuted feature importance results 

    """
    # list feature names
    feature_names = {'behavior': [0,4]}                    
    # for each lookback period
    for t in range(0,(test_X.shape[1])):
        for key in feature_names:
            eval_X_copy = perm_feat(feature_names[key][0], feature_names[key][1], t, test_X)
            y_pred, y_prob, y_predmax = perm_assess(model, eval_X_copy)
            drow = algo_var(test_y, test_dft, y_label, y_pred, y_predmax, y_prob, key, t, prob = prob)
            df = pd.concat([df, drow], axis = 0, ignore_index = True)  
            print (key + ' timestep: ' + str(t))
    df['ID'] = seed            
    maxval = int(df.shape[1])-1                  
    df = df.iloc[:,np.r_[maxval,0:maxval]] # make sure ID is first column
    df.to_csv(path + name + '.csv')
    return df

def perm_full(model, df, test_X, test_dft, test_y, y_label, seed, name, path, prob = True):
    """
    permutate each feature in each timestep and recalcualte performance metrics

    Parameters
    ----------
    model : tensor network,
        model
    df : dataframe,
        original performance metrics (non-permuted)
    test_X : array,
        testing features
    test_dft : array,
        testing deterministic features
    test_y : array,
        testing targets (one-hot encoded)
    y_label : array,
        testing targets, labelled 
    seed : int,
        random seed    
    name : str, 
        filename to save
    path: str,
        path directory to save
    prob: bool, optional
        True when prediction draw from probilities, False when prediction taken as highest probability. The default is True
 
    Returns
    -------
    df : dataframe,
        permuted feature importance results 

    """
    # list feature names
    feature_names = {'behavior': [0,4],
                     'since_rest': [4,5],
                     'since_feed': [5,6],
                     'since_travel': [6,7],
                     'sex': [7,8],
                     'reproduction': [8,12], 
                     'fragment': [12,13], 
                     'rain': [13,14], 
                     'temperature': [14,15], 
                     'flower_count': [15,16],
                     'fruit_count': [16,17],
                     'years': [17,18], 
                     'minutes': [18,20],
                     'doy': [20,22], 
                     'adults': [22,23], 
                     'infants': [23,24], 
                     'juveniles': [24,25], 
                     'individual_continuity':[25,26],
                     'length':[26,27], 
                     'position':[27,28]}
                    
    # for each lookback period
    for t in range(0,(test_X.shape[1])):
        for key in feature_names:
            eval_X_copy = perm_feat(feature_names[key][0], feature_names[key][1], t, test_X)
            y_pred, y_prob, y_predmax = perm_assess(model, eval_X_copy)
            drow = algo_var(test_y, test_dft, y_label, y_pred, y_predmax, y_prob, key, str(t), prob = prob)
            df = pd.concat([df,drow], axis = 0, ignore_index=True)  
            print (key + ' timestep: ' + str(t))       
    df['ID'] = seed            
    maxval = int(df.shape[1])-1                  
    df = df.iloc[:,np.r_[maxval,0:maxval]] # make sure ID is first column
    df.to_csv(path + name + '.csv')
    return df

def perm_internal(model, df, test_X, test_dft, test_y, y_label, seed, name, path, prob = True):
    """
    permutate each internal feature in each timestep and recalcualte performance metrics

    Parameters
    ----------
    model : tensor network,
        model
    df : dataframe,
        original performance metrics (non-permuted)
    test_X : array,
        testing features
    test_dft : array,
        testing deterministic features
    test_y : array,
        testing targets (one-hot encoded)
    y_label : array,
        testing targets, labelled 
    seed : int,
        random seed
    name : str, 
        filename to save
    path: str,
        path directory to save
    prob: bool, optional
        True when prediction draw from probilities, False when prediction taken as highest probability. The default is True
 
    Returns
    -------
    df : dataframe,
        permuted feature importance results 

    """
    # list feature names
    feature_names = {'behavior': [0,4],
                     'since_rest': [4,5],
                     'since_feed': [5,6],
                     'since_travel': [6,7],
                     'sex': [7,8],
                     'reproduction': [8,12]}
                    
    # for each lookback period
    for t in range(0,(test_X.shape[1])):
        for key in feature_names:
            eval_X_copy = perm_feat(feature_names[key][0], feature_names[key][1], t, test_X)
            y_pred, y_prob, y_predmax = perm_assess(model, eval_X_copy)
            drow = algo_var(test_y, test_dft, y_label, y_pred, y_predmax, y_prob, key, str(t), prob = prob)
            df = pd.concat([df,drow], axis = 0, ignore_index=True)  
            print (key + ' timestep: ' + str(t))
    df['ID'] = seed            
    maxval = int(df.shape[1])-1                  
    df = df.iloc[:,np.r_[maxval,0:maxval]] # make sure ID is first column
    df.to_csv(path + name + '.csv')
    return df

def perm_external(model, df, test_X, test_dft, test_y, y_label, seed, name, path, prob = True):
    """
    permutate each external feature in each timestep and recalcualte performance metrics

    Parameters
    ----------
    model : tensor network,
        model
    df : dataframe,
        original performance metrics (non-permuted)
    test_X : array,
        testing features
    test_dft : array,
        testing deterministic features
    test_y : array,
        testing targets (one-hot encoded)
    y_label : array,
        testing targets, labelled 
    seed : int,
        random seed
    name : str, 
        filename to save
    path: str,
        path directory to save
    prob: bool, optional
        True when prediction draw from probilities, False when prediction taken as highest probability. The default is True
 
    Returns
    -------
    df : dataframe,
        permuted feature importance results 

    """
    # list feature names
    feature_names = {'reproduction': [0,4], 
                     'fragment': [4,5], 
                     'rain': [5,6], 
                     'temperature': [6,7], 
                     'flower_count': [7,8],
                     'fruit_count': [8,9],
                     'years': [9,10], 
                     'minutes': [10,12],
                     'doy': [12,14], 
                     'adults': [14,15], 
                     'infants': [15,16], 
                     'juveniles': [16,17]}
                    
    # for each lookback period
    for t in range(0,(test_X.shape[1])):
        for key in feature_names:
            eval_X_copy = perm_feat(feature_names[key][0], feature_names[key][1], t, test_X)
            y_pred, y_prob, y_predmax = perm_assess(model, eval_X_copy)
            drow = algo_var(test_y, test_dft, y_label, y_pred, y_predmax, y_prob, key, str(t), prob = prob)
            df = pd.concat([df,drow], axis = 0, ignore_index=True)  
            print (key + ' timestep: ' + str(t))
            
    df['ID'] = seed            
    maxval = int(df.shape[1])-1                  
    df = df.iloc[:,np.r_[maxval,0:maxval]] # make sure ID is first column
    df.to_csv(path + name + '.csv')
    return df

def perm_importance(model, params, df, test_X, test_dft, test_y, y_label, seed, name, path, prob = True):
    """
    wrapper to run permutation feature importance functions

    Parameters
    ----------
    model : tensor network,
        model
    params: dict,
        model parameters
    df : dataframe,
        original performance metrics (non-permuted)
    test_X : array,
        testing features
    test_dft : array,
        testing deterministic features
    test_y : array,
        testing targets (one-hot encoded)
    y_label : array,
        testing targets, labelled 
    seed : int,
        random seed
    name : str, 
        filename to save
    path: str,
        path directory to save results
    prob: bool, optional
        True when prediction draw from probilities, False when prediction taken as highest probability. The default is True
    
    Raises
    ------
    Exception
        something other than 'full','behavior','internal','external' designated in params['predictor']
    
    Returns
    -------
    rnn_perm : dataframe, 
        permutation feature importance results

    """     
    if params['predictor'] == 'full':
        rnn_perm = perm_full(model, df, test_X, test_dft, test_y, y_label, seed, name + '_perm_importance', path, prob = prob)
    elif params['predictor'] == 'behavior':
        rnn_perm = perm_behavior(model, df, test_X, test_dft, test_y, y_label, seed, name + '_perm_importance', path, prob = prob)
    elif params['predictor'] == 'internal':
        rnn_perm = perm_internal(model, df, test_X, test_dft, test_y, y_label, seed, name + '_perm_importance', path, prob = prob)
    elif params['predictor'] == 'external':
        rnn_perm = perm_external(model, df, test_X, test_dft, test_y, y_label, seed, name + '_perm_importance', path, prob = prob)
    else:
        raise Exception ('invalid predictor set')
    return rnn_perm

def eval_pipeline(train, test, params, path, prob = True):
    '''
    Run model, evaluate performance and conduct permutation feature importance analysis    
    
    Parameters
    ----------
    train : dataframe,
        training data
    test : dataframe,
        testing data
    params : dict,
        model parameters
    path : str,
        directory to save file.
    prob: bool, optional
        True when prediction draw from probilities, False when prediction taken as highest probability. The default is True
 

    Raises
    ------
    Exception
        something other than 'VRNN' or 'ENDE' designated as params['atype']
    

    Returns
    -------
    rnn_perm : dataframe, 
        permutation feature importance results

    '''
    rannum = random.randrange(1,200000,1) # draw random seed
    random.seed(rannum) # assign seed 
    # format training and testing data
    train_X, train_y, train_dft, test_X, test_y, test_dft = train_test_format(train, test, params)
    
    targets = 4 # set number of targets (4 behavior classes)
    features = test_X.shape[2] # get number of features 
    
    model = build_model_func(params, features, targets) # build model
    
    # assign class weights
    weights = dict(zip([0,1,2,3], 
                        [params['weights_0'], 
                        params['weights_1'], 
                        params['weights_2'], 
                        params['weights_3']]))
    
    # assign the callback and weight type based on the model type
    if params['atype'] == 'VRNN':
        class_weights = weights # assign class weights as weights
        sample_weights = None
    elif params['atype'] == 'ENDE':
        class_weights = None 
        sample_weights = get_sample_weights(train_y, weights) # generate the formatted sample weights 
    else:
        raise Exception ('invalid model type')
        
    # fit the model 
    history = model.fit(train_X, 
                        train_y,
                        epochs = params['max_epochs'], 
                        batch_size = params['batch_size'],
                        sample_weight = sample_weights,
                        class_weight = class_weights,
                        verbose = 2,
                        shuffle=False)
    print('model fitted')
    
    # get predictions
    y_prob = model.predict(test_X)
    y_label = to_label(test_y) # observed target
    y_pred = to_label(y_prob, prob = True) # predicted target
    y_predmax = to_label(y_prob, prob = False) # predicted target
    
    # merge data together, will be used to get daily distributions
    results_df = np.column_stack((test_dft, y_label, y_pred, y_predmax, y_prob))
    
    # column names for merged data
    names = ['sex', 'gestation', 'lactation', 'mating', 'nonreproductive',
        'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count',
        'years', 'minutes_sin', 'minutes_cos', 'doy_sin', 'doy_cos', 'adults',
        'infants', 'juveniles', 'individual_continuity', 'length', 'position',
        'obs','pred','predmax','feed_prob','rest_prob','social_prob','travel_prob']
     
    # convert to dataframe
    pred_df = DataFrame(results_df, columns = names)
    
    # assign filename
    if params['loss'] == True:
        name = params['atype'] + '_' + params['mtype'] + '_' + params['predictor'] + '_catloss_' + str(rannum)
    else:
        name = params['atype'] + '_' + params['mtype'] + '_' + params['predictor'] + '_f1loss_' + str(rannum)
   
    # save subset of results
    pred_df.loc[:, ['obs','pred','predmax','feed_prob','rest_prob','social_prob','travel_prob', 'years', 'doy_sin','doy_cos']].to_csv(path + name + '_predictions.csv')
     
    print('predictions generated and saved')
    
    # evaluate performance of model
    df = algo_var(test_y, test_dft, y_label, y_pred, y_predmax, y_prob, 'original', prob = prob)
     
    # run permutation feature importance
    rnn_perm = perm_importance(model, 
                               params, 
                               df, 
                               test_X, 
                               test_dft, 
                               test_y, 
                               y_label, 
                               seed = rannum, 
                               name = name, 
                               path = path,
                               prob = prob)
    print('permutation feature importance completed')
    
    del history, model # ensure memory is freed up
    return rnn_perm

def pi_plot(df, metrics):
    '''
    Plot permutation importance metrics
    Parameters
    ----------
    df : dataframe,
        permutation importance dataframe
    metric : list,
        metric to plot to measure permutation importance

    Returns
    -------
    fig: plot,
        figure

    '''
    features = df['feature'].unique() # get unique features
    features = np.delete(features,0) # get rid of original 
    nf = len(features) # number of features 
    nm = len(metrics) # number of metrics
    counter =1
    fig, axs = plt.subplots(nf, nm, figsize = (3*nm, 3*nf))
    for i in range(nf):
        for metric in metrics:
            plt.subplot(nf,nm, counter)
            sub_df = df[df['feature'] == features[i]]
            plt.bar(x = (sub_df['lookback']+1)*-1, 
                    height = sub_df[metric])
            plt.axhline(y = df[df['feature'] == 'original'][metric][0], 
                         linestyle = '--', 
                         linewidth =1, 
                         color = 'red')
            plt.title(features[i]) # add title
            plt.xlabel('lookbacks prior')
            plt.ylabel(metric)
            counter+=1
    fig.tight_layout()  
    plt.show()
    
    return fig

#####################
#### Null models ####
#####################
def null_mod(train_y, test_y, test_dft, name):
    """
    Generate null0 model predictions, which are drawn from the overall behavioral frequency distributions
    Assess prediction performance

    Parameters
    ----------
    train_y: array,
        training targets (one-hot encoded)
    test_y : array,
        testing targets (one-hot encoded)
    test_dft : array,
        testing deterministic features
    name : str,
        model name 

    Returns
    -------
    drow : dataframe,
        performance metrics

    """
    seed = random.randrange(1,200000,1) # draw random seed
    random.seed(seed) # assign seed 
    
    # generate testing target 
    y_label = to_label(test_y, prob = False)
    
    # generate training targets
    train_ylab = to_label(train_y)
    
    # calculate activity distribition
    train_prop = np.unique(train_ylab, return_counts = True)
    train_prop = train_prop[1]/len(train_ylab)
    
    # duplicate act dist probabilities for the number of testing targets
    y_prob = np.repeat([train_prop], 
                       repeats = len(y_label), 
                       axis = 0)
    y_pred = to_label(y_prob, prob = True) # predicted target
    y_predmax = to_label(y_prob, prob = False) # predicted target
    
    # evaluate act dist predictions
    drow = algo_var(test_y, 
                    test_dft, 
                    y_label, 
                    y_pred, 
                    y_predmax, 
                    y_prob, 
                    name, 
                    lookback = -99, 
                    prob = True) 
    
    # reorder dataframe
    drow['ID'] = seed
    maxval = drow.shape[1]-1
    drow = drow.iloc[:, np.r_[maxval, 0:maxval]]
    return drow

def transition_matrix(train_X, train_y):
    """
    Get transition matrix from behavioral data
    Parameters
    ----------
    data : array,
        training targets (labelled)

    Returns
    -------
    M : transition matrixes 

    """
    
    transitions = train_X[:,0,0:4]
    transitions = to_label(transitions, prob = False)
    predictor = to_label(train_y, prob = False)
    
    n = 1+ max(transitions) #number of states

    # create empty shell to populate transition probs
    M = [[0]*n for _ in range(n)]
    
    # get counts of transitions
    for (i,j) in zip(transitions,predictor):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    
    # convert to dictionary
    M = {0: M[0],
          1: M[1],
          2: M[2],
          3: M[3]}
    
    return M

def markov_null(train_X, train_y, test_X, test_y, test_dft, name):
    """
    Generate markov model predictions, which are drawn from the transition likelihood between behaviors
    Assess prediction performance

    Parameters
    ----------
    train_X : array,
        training features
    train_y : array,
        training targets (one-hot encoded)
    test_X : array,
        testing features
    test_y : array,
        testing targets (one-hot encoded)
    test_dft : array,
        testing deterministic features
    name : str,
        model name 

    Returns
    -------
    drow : dataframe,
        performance metrics

    """
    seed = random.randrange(1,200000,1) # draw random seed
    random.seed(seed) # assign seed 
    
    train_mat = transition_matrix(train_X, train_y) # generate transition matrix 
    
    # generate testing labels    
    y_label = to_label(test_y, prob = False)
    
    # extract predictors (prior behavior)
    predictors = test_X[:, 0, 0:4]
    predictors = to_label(predictors, prob = False)
    
    # generate probabilities for the each target prediction 
    y_prob = [train_mat[key] for key in predictors] 
    y_prob = np.array(y_prob)
    y_pred = to_label(y_prob, prob = True) # predicted target
    y_predmax = to_label(y_prob, prob = False) # predicted target
   
    # evaluate act dist predictions
    drow = algo_var(test_y, 
                    test_dft, 
                    y_label, 
                    y_pred, 
                    y_predmax, 
                    y_prob, 
                    name, 
                    lookback = -99, 
                    prob = True) 
    
    # reorder dataframe
    drow['ID'] = seed
    maxval = drow.shape[1]-1
    drow = drow.iloc[:, np.r_[maxval, 0:maxval]]
    return drow
