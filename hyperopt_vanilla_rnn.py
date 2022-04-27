# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet
"""
# Load libraries
import numpy as np
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv, DataFrame, concat
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
import time
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
import joblib
import seaborn as sns

# import datafile
dataset =  read_csv('data.csv', header = 0, index_col = 0)

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

def monitoring_plots(result, early_stopping = None):
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

def eval_f1_iter(model, params, train_X, train_y, test_X, test_y, patience=50, max_epochs = 300, atype = 'VRNN', n = 1):
    """
    Fit and evaluate model n number of times. Get the average of those runs

    Parameters
    ----------
    model : model
    params : hyperparameters
    train_X : training features
    train_y :  training targets
    test_X : testing features
    test_y : testing targets
    patience: early stopping patience value
    atype: architecture type (VRNN or ENDE)

    Returns
    -------
    eval_run : metrics for each iteration
    avg_val: average of the metrics average: val_f1, val_loss, train_f1, train_loss 
    """
    # assign the weights 
    weights = dict(zip([0,1,2,3], [params['weights_0'], params['weights_1'], params['weights_2'], params['weights_3']]))
    # assign the callback and weight type based on the model type
    if atype == 'VRNN':
        early_stopping = EarlyStopping(patience= patience, monitor='val_f1_score', mode = 'max', restore_best_weights=True, verbose=0)
        class_weights = weights # assign class weights as weights
        sample_weights = None
    else:
        early_stopping = F1EarlyStopping(validation_data=[test_X, test_y], train_data=[train_X, train_y], patience= patience)
        class_weights = None 
        total = sum(weights.values()) # get the sum of the weights to normalize
        sample_weights = {ky: val / total for ky, val in weights.items()} # get the sample weight values
        sample_weights = get_sample_weights(train_y, weights) # generate the formatted sample weights 
    eval_run = [] # create empty list for the evaluations
    for i in range(n): # for each iteration
        # fit the model 
        history = model.fit(train_X, train_y, 
                            epochs = max_epochs, 
                            batch_size = params['batch_size'],
                            verbose = 2,
                            shuffle=False,
                            validation_data = (test_X, test_y),
                            sample_weight = sample_weights,
                            class_weight = class_weights,
                            callbacks = [early_stopping])
        # pull out monitoring metrics
        if len(history.history['loss']) == max_epochs:
            params['epochs'] = max_epochs # assign the epochs to the maximum epochs
            val_loss = history.history['val_loss'][-1] # get the last val loss
            train_loss = history.history['loss'][-1] # get the last train loss
            if atype == 'ENDE': # if the model is an encoder-decoder
                f1 = early_stopping.val_f1s[-1] # then pull the last validation f1 from the early stopping metric
                train_f1 = early_stopping.train_f1s[-1] # pull the last training f1 from the early stopping metric
            else: # otherwise 
                f1 = history.history['val_f1_score'][-1] # pull the last validation f1 from the history
                train_f1 = history.history['f1_score'][-1] # pull the last train f1 from the history
        else: # otherwise if early stopping was activate
            params['epochs'] = len(history.history['loss'])-patience # assign stopping epoch as the epoch before no more improvements were seen in the f1 score
            val_loss = history.history['val_loss'][-patience-1] # assign validation loss from the stopping epochs
            train_loss = history.history['loss'][-patience-1] # assign trainn loss from the stopping epochs
            if atype == 'ENDE': # if the model is the encoder-decoder
                f1 = early_stopping.val_f1s[-patience-1] # assign the validation f1 from the stopping epoch through the early stopping 
                train_f1 = early_stopping.train_f1s[-patience-1] # assign the training f1 from the stopping epoch through the early stopping 
            else: # otherwise
                f1 = history.history['val_f1_score'][-patience-1] # assign the training f1 from the stopping epoch
                train_f1 = history.history['f1_score'][-patience-1] # assign the training f1 from the stopping epoch
        eval_run.append([f1,val_loss,train_f1,train_loss])
    avg_val = np.mean(eval_run,axis=0)
    return eval_run, avg_val[0], avg_val[1], avg_val[2], avg_val[3]

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

# hyperoptimization functions
def run_trials(filename, objective, space, rstate, initial = 20, trials_step = 1):
    """
    Run trials indefinitely until manually stopped
    
    Parameters
    ----------
    filename : trial filename
    objective : objective
    space : parameters
    initial: initial number of trials, should be >= 20 
    rstate: set random state for consistency across trials
    trials_steps: how many additional trials to do after loading saved trials.
    
    Returns
    -------
    None.

    """
    max_trials = initial  # set the initial trials to run (should be at least 20, since hyperopt selects parameters randomly before 20 trials)
    try:  # try to load an already saved trials object, and increase the max
        trials = joblib.load(filename) # load file 
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step # increase the max_evals value
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # if trial file cannot be found
        trials = Trials() # create a new trials object
    # run the search
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_trials, trials=trials, rstate = np.random.RandomState(rstate))
    print("Best:", best)
    print("max_evals:", max_trials)
    joblib.dump(trials, filename) # save the trials object
  
# Visualization functions
def trial_correg_plots(trials):
    """
    Format hyperopt trials into dataframe
    Visualize correlation plots, training v. validation losses, training v. validation f1

    Parameters
    ----------
    trials : hyperopt trials
    Returns : correlation plot for all result metrics
    -------
    df: dataframe of the trial results

    """
    df = pd.DataFrame(trials.results) # extract the trial results and make into dataframe
    df1 = df.params.to_list() # convert the parameter dictionarys to list to convert into dataframe as well
    df1 = pd.DataFrame(df1) # convert into dataframe
    df2 = df1.architecture.to_list()
    df2 = pd.DataFrame(df2) # convert into dataframe
    df1.drop(columns = ['architecture'], axis = 1)
    df = concat([df,df1,df2], axis = 1) # concatenate the two trial result dataframes
    df.drop(columns = ['status','params','architecture'], axis = 1, inplace = True) # drop unnecessary columns
    if 'val_f1' not in df.columns: # if this column is missing
        df.rename(columns={'loss':'val_f1'}, inplace=True) # then rename the loss column to the val_f1 column
        df['val_f1'] = df['val_f1']*-1 # multiple by -1 to get true value
    df = df.iloc[0:1000,:]
    vcorr = df.corr() # calculate the correlatio
    fig, axis = pyplot.subplots(3,1,figsize=(5,10))
    pyplot.subplot(3, 1, 1) # divide the plot space 
    sns.heatmap(vcorr, 
                xticklabels=vcorr.columns,
                yticklabels=vcorr.columns,
                cmap = 'vlag')
    pyplot.subplot(3, 1, 2) # divide the plot space 
    sns.regplot(df['val_loss'], df['train_loss'], fit_reg = True,color = 'orange') # plot the relationship between val loss and train loss
    pyplot.subplot(3, 1, 3) # divide the plot space 
    sns.regplot(df['val_f1'], df['train_f1'], fit_reg = True, color = 'green') # plot the relationship between val f1 and train f1
    fig.tight_layout()  
    return df

def hopt_comp(seeds, file_prefix, atype):
    """
    Convert the trials to dataframe and subset out the trials with high f1 scores for further visualization/analysis
    Monitor convergence of the hyperparameter optimization trials

    Parameters
    ----------
    seeds : list of 3 seeds
    file_prefix : file prefix name
    atype : architecture type

    Returns
    -------
    Dictionary with the three trial dataframes

    """
    # load trials
    rnn_0 = joblib.load(file_prefix + str(seeds[0]) + '.pkl')
    rnn_1 = joblib.load(file_prefix + str(seeds[1]) + '.pkl')
    rnn_2 = joblib.load(file_prefix + str(seeds[2]) + '.pkl')
    # convert trials to dataframe and get correlation, loss and f1 plots 
    rnn0_df = trial_correg_plots(rnn_0)
    rnn1_df = trial_correg_plots(rnn_1)
    rnn2_df = trial_correg_plots(rnn_2)
    # save dataframes
    rnn0_df.to_csv(atype+str(seeds[0])+'.csv')
    rnn1_df.to_csv(atype+str(seeds[1])+'.csv')
    rnn2_df.to_csv(atype+str(seeds[2])+'.csv')
    
    # plot the validation loss across trials run over time to see if the search is converging on hyperparameters that produce the higher f1 scores
    fig, axis = pyplot.subplots(1,3,figsize=(16,4))
    pyplot.subplot(1, 3, 1) # divide the plot space 
    sns.lineplot(x = rnn0_df.index, y = rnn0_df["val_f1"])
    pyplot.subplot(1, 3, 2) # divide the plot space 
    sns.lineplot(x = rnn1_df.index, y = rnn1_df["val_f1"])
    pyplot.subplot(1, 3, 3) # divide the plot space 
    sns.lineplot(x = rnn2_df.index, y = rnn2_df["val_f1"])
    fig.tight_layout()  

    return{'rnn_0': rnn0_df,
           'rnn_1': rnn1_df,
           'rnn_2': rnn2_df}
 
def sub_comp(df_dict, threshold = 0.4):
    """
    Get out the trials whose f1 >= threshold, map the correlation plot and map the categorical variable selection

    Parameters
    ----------
    df_dict : dictionary that contains the 3 trials to be compared
    threshold : minimum value for the validation f1 score that should be examined 

    Returns
    -------
    Dictionary with the 3 trial dataframes with validation f1 score above the specified threshold

    """
    # extract the dataframes 
    rnn0_df = df_dict['rnn_0']
    rnn1_df = df_dict['rnn_1']
    rnn2_df = df_dict['rnn_2']
    
    # subset out dataset with validation f1 scores above a certain threshold
    rnn0_sub = rnn0_df[rnn0_df['val_f1'] >= threshold] 
    rnn1_sub = rnn1_df[rnn1_df['val_f1'] >= threshold] 
    rnn2_sub = rnn2_df[rnn2_df['val_f1'] >= threshold] 
    
    # for the categorical hyperparameter examine a barplot comparing the choices
    fig, axis = pyplot.subplots(1,3,figsize=(16,4))
    pyplot.subplot(1, 3, 1) # divide the plot space 
    rnn0_sub['mtype'].value_counts().plot(kind='bar')
    pyplot.subplot(1, 3, 2) # divide the plot space 
    rnn1_sub['mtype'].value_counts().plot(kind='bar')
    pyplot.subplot(1, 3, 3) # divide the plot space 
    rnn2_sub['mtype'].value_counts().plot(kind='bar')
    
    # get correlation plots for f1 values >= 0.4
    fig, axis = pyplot.subplots(1,3,figsize=(16,4))
    pyplot.subplot(1, 3, 1) # divide the plot space 
    sns.heatmap(rnn0_sub.corr(), 
                xticklabels=rnn0_sub.corr().columns,
                yticklabels=rnn0_sub.corr().columns,
                cmap = 'vlag')
    pyplot.subplot(1, 3, 2) # divide the plot space 
    sns.heatmap(rnn1_sub.corr(), 
                xticklabels=rnn1_sub.corr().columns,
                yticklabels=rnn1_sub.corr().columns,
                cmap = 'vlag')
    pyplot.subplot(1, 3, 3) # divide the plot space 
    sns.heatmap(rnn2_sub.corr(), 
                xticklabels=rnn2_sub.corr().columns,
                yticklabels=rnn2_sub.corr().columns,
                cmap = 'vlag')
    fig.tight_layout()  
    
    print([len(rnn0_sub),len(rnn1_sub), len(rnn2_sub)]) # get out number of trials that were above the threshold
    return{'rnn_0': rnn0_sub,
           'rnn_1': rnn1_sub,
           'rnn_2': rnn2_sub}
    
def kde_comp(groups, sub_df, ymax):
    """
    Get kernel density plots for each hyperparameter

    Parameters
    ----------
    groups : the column indices for the hyperparameters of interest
    sub_df : dictionary for the 3 trials being compared. The dictionary should contain the trials in which val_f1 >= 0.4
    ymax : maximum y limit for the graph

    Returns
    -------
    Outputs the kernel density graphs

    """
    n = len(groups)
    groups = sub_df['rnn_0'].columns[[groups]]
    # repeat this for each group  
    fig, axis = pyplot.subplots(n,3,figsize=(16,12)) # set the plot space dimensions
    i = 1 # start counter
    for group in groups:
        pyplot.subplot(n, 3, i) # divide the plot space 
        g = sns.kdeplot(sub_df['rnn_0'][sub_df['rnn_0'][group].notnull()][group],sub_df['rnn_0'][sub_df['rnn_0'][group].notnull()]['val_f1'], 
                    shade = True, shade_lowest=False, legend = True, color = 'purple') # plot values that don't have nulls for the hyperparameter of interest against the val f1
        g.set(ylim=(0.4, ymax))
        pyplot.subplot(n, 3, i+1) # divide the plot space 
        g1 = sns.kdeplot(sub_df['rnn_1'][sub_df['rnn_1'][group].notnull()][group],sub_df['rnn_1'][sub_df['rnn_1'][group].notnull()]['val_f1'], 
                    shade = True, shade_lowest=False, legend = True, color = 'purple') # plot values that don't have nulls for the hyperparameter of interest against the val f1
        g1.set(ylim=(0.4, ymax))
        pyplot.subplot(n, 3, i+2) # divide the plot space 
        g2 = sns.kdeplot(sub_df['rnn_2'][sub_df['rnn_2'][group].notnull()][group],sub_df['rnn_2'][sub_df['rnn_2'][group].notnull()]['val_f1'], 
                    shade = True, shade_lowest=False, legend = True, color = 'purple') # plot values that don't have nulls for the hyperparameter of interest against the val f1
        g2.set(ylim=(0.4, ymax))
        i+=3 # increase the counter
    fig.tight_layout()  
  
def hyp_rnn_nest(params, features, targets):
    """
    Model for hyperparameter optimizer.
    Basic RNN for single timestep output prediction (one-to-one or many-to-one)
    With nested architecture
    Parameters
    ----------
    params: hyperparameters 
    features: number of features
    targets: number of targets
    Returns
    -------
    model : model
    """ 
    lookback = params['lookback'] # extract lookback
    model = Sequential() # create an empty sequential shell 
    model.add(Masking(mask_value = -1, input_shape = (lookback, features), name = 'Masking')) # add a masking layer to tell the model to ignore missing values
    if params['mtype']=='LSTM': # if the model is an LSTM
        model.add(LSTM(units =params['neurons_n'], input_shape = (lookback,features), return_sequences= params['architecture']['return_seq'], name = 'LSTM')) # set the RNN type
        model.add(Dropout(rate= params['drate'])) # add another drop out
        if params['architecture']['stacked_layers'] == 0: # if there are no stacked layers
            for i in range(params['architecture']['hidden_layers']): # increase the model complexity through adding more hidden layers 
                model.add(Dense(units = params['architecture']['hidden_n'+str(i)], activation = 'relu', kernel_initializer =  'he_uniform')) # add dense layer
                model.add(Dropout(rate= params['drate'])) # add dropout rate
        else: # otherwise increase the complexity of the model through stacking layers
            stacked_layers = params['architecture']['stacked_layers']-1 # extract out how many stacked layers need to be added w/ return_seq = True
            for i in range(stacked_layers): # iterate through stacked layers 
                model.add(LSTM(units =params['architecture']['stacked_n'+str(i)], return_sequences= params['architecture']['return_seq'])) # set the RNN type
                model.add(Dropout(rate= params['drate'])) # add dropout
            model.add(LSTM(units =params['architecture']['stacked_n'+str(stacked_layers)])) # add the last stacked layer with return_seq = false
            model.add(Dropout(rate= params['drate'])) # add another drop out
    else: # the model is a GRU, set architecture accordingly
        model.add(GRU(units =params['neurons_n'], input_shape = (lookback,features), return_sequences= params['architecture']['return_seq'],name = 'GRU')) # set the RNN type
        model.add(Dropout(rate= params['drate'])) # add another drop out
        if params['architecture']['stacked_layers'] == 0: # if there are no stacked layers to add
            for i in range(params['architecture']['hidden_layers']):# increase complexity through hidden layers
                model.add(Dense(units = params['architecture']['hidden_n'+str(i)], activation = 'relu', kernel_initializer = 'he_uniform')) # add dense layer
                model.add(Dropout(rate= params['drate'])) # add dropout layer
        else: # otherwise increase complexity through stacking
            stacked_layers = params['architecture']['stacked_layers']-1 # calculate how may stacks to add with return_seq = True
            for i in range(stacked_layers): # iterate through stacked layers 
                model.add(GRU(units =params['architecture']['stacked_n'+str(i)], return_sequences= params['architecture']['return_seq'])) # set the RNN type
                model.add(Dropout(rate= params['drate'])) # add dropout
            model.add(GRU(units =params['architecture']['stacked_n'+str(stacked_layers)])) # add last stacked layer with return_seq = False
            model.add(Dropout(rate= params['drate'])) # add another dropout
    model.add(Dense(units = targets, activation = "softmax", name = 'Output')) # add output layer
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = params['learning_rate']), metrics = [F1Score(num_classes=4, average = 'macro')]) # compile the model
    return model 

def hyperoptimizer_vrnn(params):
    """
    hyperparameter optimizer objective function

    Parameters
    ----------
    params : hyperparameter search space

    Returns
    -------
    dict
        loss: loss value to optimize through minimization (i.e., -validation f1)
        status: default value for hyperopt
        params: the hyperparameter values being tested
        val_loss: validation loss
        train_loss: training loss
        train_f1: training f1

    """
    features=26 # set features
    targets=4 # set targets
    model = hyp_rnn_nest(params, features, targets) # build model based on hyperparameters
    train, test = split_dataset(dataset, 2015) # split the data
    train_X, train_y, train_dft = to_supervised(data = train, TID = train['TID'], window = 1, lookback = params['lookback'], n_output=params['n_output']) # format training data
    test_X, test_y, test_dft = to_supervised(data = test, TID = test['TID'],window = params['n_output'], lookback = params['lookback'], n_output = params['n_output']) # format testing data
    _, f1_value,val_loss,train_f1, train_loss = eval_f1_iter(model, params, train_X, train_y, test_X, test_y, patience = 30, atype ='VRNN', max_epochs = 100, n=1) # fit model and extract monitoring metrics
    print('Best validation for trial:', f1_value) # print the validation score
    return {'loss': -f1_value,
            'status': STATUS_OK,  
            'params': params,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'train_f1':train_f1}   

# hyperparameter space
space = {'drate'                  : hp.uniform('drate',0.5,0.9),
            'neurons_n'           : scope.int(hp.quniform('neurons_n',1,50,1)),
            'n_output'            : 1,
            'learning_rate'       : hp.loguniform('learning_rate',np.log(0.00001),np.log(0.001)),
            'architecture'        : hp.choice('architecture',
                                    [
                                     {'hidden_layers': 0,
                                      'stacked_layers': scope.int(hp.quniform('stacked_layers',1,2,1)),
                                      'return_seq': True,
                                      'stacked_n0' : scope.int(hp.quniform('stacked_n0',1,100,1)),
                                      'stacked_n1' : scope.int(hp.quniform('stacked_n1',1,100,1))
                                      },
                                     {'hidden_layers': scope.int(hp.quniform('layers',0,2,1)),
                                      'stacked_layers': 0,
                                      'return_seq': False,
                                      'hidden_n0' : scope.int(hp.quniform('hidden_n0',1,100,1)),
                                      'hidden_n1' : scope.int(hp.quniform('hidden_n1',1,100,1))
                                      },]),
            'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
            'epochs'                 : 0,
            'batch_size'             : scope.int(hp.quniform('batch_size',144,512,12)),
            'weights_0'              : hp.uniform('weights_0',20,60),
            'weights_1'              : hp.uniform('weights_1',0.0001,30),
            'weights_2'              : hp.uniform('weights_2',40,200),
            'weights_3'              : hp.uniform('weights_3',30,80),
            'mtype'                  : hp.choice('mtype',['GRU','LSTM'])
        }
        
# run trials for certain number of evals
trials = Trials() # create an empty Trials object to populate
start_time = time.time() # generate the start time to keep track of run time
# assign the fitting objective, the algorithm , the number of trials, and save each trial, added early stopping here with pateince of 20
best = fmin(hyperoptimizer_vrnn, space, algo=tpe.suggest, max_evals = 2, trials = trials,early_stop_fn=no_progress_loss(20), rstate = np.random.RandomState(20)) # output the hyperparameter set for the best model (min losses)
print('took', (time.time()-start_time)/3600, ' hours') # print the time lapsed 
print(best) # print the best hyperparameter set

joblib.dump(trials, 'vanilla_rnn_trials.pkl') # saving trial

# run hyperparameter search indefinitely and manually stop whenever
g = 1
while g < 1000:
    run_trials(filename = 'vanilla_rnn_trials.pkl',objective =hyperoptimizer_vrnn, space =space, rstate =6, initial=25, trials_step=2)
    g += 1

# load hyperopt trials
vrnn_1 = joblib.load('vanilla_rnn_trials_seed1.pkl')
vrnn_2 = joblib.load('vanilla_rnn_trials_seed2.pkl')
vrnn_3 = joblib.load('vanilla_rnn_trials_seed3.pkl')

# plot trial results
vrnn_dict = hopt_comp([1,2,3], 'vanilla_rnn_vv_trials_seed','vrnn')
vrnn_sub = sub_comp(vrnn_dict)
kde_comp([4,5,6,7,8],vrnn_sub,0.45)
kde_comp([11,12,13,14,15],vrnn_sub,0.45)
