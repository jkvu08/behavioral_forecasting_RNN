# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 02:30:09 2021

@author: Jannet

"""
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
import os, time
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import os, random

# Load and Prepare Dataset
path = 'filepath/' # set working directory
# read in behavioral 
data =  read_csv(path+'data.csv', header = 0, index_col = 0) 

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

# Permutation analysis
def pi_info(cm, report, feature, lookback):
    """
    extract information from the confusion matrix and classification report

    Parameters
    ----------
    cm : confusion matrix
    report : report
    feature : feature number
    lookback : lookback timestep (# timesteps backwards)

    Returns
    -------
    data for row to add to dataframe

    """
    # indices assigned according to 4 prediction classes
    datarow = [feature,lookback,report.iloc[4,0]] # add lookback and feature and accuracy (3)
    datarow.extend(report.iloc[5,0:3].tolist()) # overall precision, recall and f1 score (3)
    datarow.extend(report.iloc[0:4,2].tolist()) # add categorical f1 score (4)og_values.extend(gru_f1['report'].iloc[0:4,0].tolist()) # add categorical precision (4)
    datarow.extend(report.iloc[0:4,0].tolist()) # add categorical precision (4)
    datarow.extend(report.iloc[0:4,1].tolist()) # add categorical recall (4)
    conmat = np.reshape(cm,(1,16)) # add cofusion matrix values (16)
    datarow.extend(conmat.ravel().tolist())
    return(datarow)

def perm_feat(start_index, end_index, t, eval_X):
    """
    Permute the feature(s) for the permutation feature importance assessment
    Parameters
    ----------
    start_index : first indices of for the columns that pertain to the categorical varaible
    end_index : 1+ last indices of for the columns that pertain to the categorical varaible
    t : is lookback timestep being evaluated
    eval_X : features dataset

    Returns
    -------
    None.

    """
    eval_X_copy = np.copy(eval_X) # make copy of the original training features
    # first deal with the behavior variable (categorical so all behavior onehot need to be shuffled in tandem)
    value = np.copy(eval_X_copy[:,t,start_index:end_index]) # make a copy of behavior columns
    eval_X_copy[:,t,start_index:end_index] = np.random.permutation(value) # permute the rows and replace the values in the copied df
    return(eval_X_copy)
    
def perm_assess(model, X_reshape, y): 
    """
    Make predictions using the the permutated feature(s). 
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
    y_pred = to_label(y_prob)
    #y_pred= y_prob.argmax(axis=-1)
    
    # get confusion matrix
    cm = confusion_matrix(y,y_pred)

    # get classification report
    class_rep = classification_report(y,y_pred, zero_division = 0, output_dict = True)
    class_rep = DataFrame(class_rep).transpose()
    
    return {'confusion_matrix': cm, 
            'report': class_rep}

# iterate throuhg all the lookbacks and features to reu
def perm_importance(model, og_cm, og_report, eval_X, y):
    '''
    Make predictions using the the permutated feature(s). 
    Parameters
    ----------
    model : fitted model
    og_cm : baseline confusion matrix (generated prior to permutating features
    eval_X : testing features dataset
    y : target data
    Returns
    -------
    dict
    '''
    # list column names 
    colnames = ['feature','lookback','accuracy','precision','recall','f1_score','f1_f','f1_r','f1_s','f1_t',
            'precision_f','precision_r','precision_s','precision_t','recall_f','recall_r','recall_s','recall_t',
            'FF','FR','FS','FT','RF','RR','RS','RT','SF','SR','SS','ST','TF','TR','TS','TT']
    # list feature names
    feature_names = ['behavior','reproduction','sex','length','position','flower_count','fruit_count',
                     'year','since_rest','since_feed','since_travel','adults','infants',
                     'juveniles','rain','temperature', 'minutes','doy']
    df = pd.DataFrame(columns = colnames) # create dataframe to keep records of permutation importance metrics
    datarow = pi_info(og_cm, og_report, 'original', np.nan) # get the baseline metrics 
    df.loc[len(df.index)]= datarow # set as first row of dataframe
    
    # for each lookback period
    for t in range(0,(eval_X.shape[1])):
        counter = 0 # start counter for feature name
        # run categorical variables first, which have to be permutated as sets
        # run for behavior
        eval_X_copy = perm_feat(0,4,t,eval_X) # permute behavior
        perm_output = perm_assess(model, eval_X_copy, y) # assess 
        datarow = pi_info(perm_output['confusion_matrix'], perm_output['report'], feature_names[counter], (eval_X.shape[1]-t)) # extract metrics
        df.loc[len(df.index)]= datarow # add new metrics to dataframe to records dataframe
        print(feature_names[counter], t) # track progress 
        counter +=1 # increase counter 
        
        # run for reproduction
        eval_X_copy = perm_feat(4,8,t,eval_X) # permute reproduction
        perm_output = perm_assess(model, eval_X_copy, y) # assess 
        datarow = pi_info(perm_output['confusion_matrix'], perm_output['report'], feature_names[counter], (eval_X.shape[1]-t)) # get datarow
        df.loc[len(df.index)]= datarow  # add to end of records dataframe
        print(feature_names[counter], t) # track progress
        counter +=1 # increase counter

        # for variables that span only 1 column
        # get indices of the single variable columns in the features dataset 
        single_var = np.r_[8:15,18:27]
        for f in single_var:
            eval_X_copy = perm_feat(f,f+1,t,eval_X) # permute the feature
            perm_output = perm_assess(model, eval_X_copy, y) # assess 
            datarow = pi_info(perm_output['confusion_matrix'], perm_output['report'], feature_names[counter], (eval_X.shape[1]-t)) # get data row
            df.loc[len(df.index)]= datarow # add new row into records dataframe
            print(feature_names[counter], t) # print to monitor progress
            counter +=1 # increase counter
        
        # permutation importance for time variables that span 2 columns
        eval_X_copy = perm_feat(14,16,t,eval_X)  # permute minutes
        perm_output = perm_assess(model, eval_X_copy, y) # assess
        datarow = pi_info(perm_output['confusion_matrix'], perm_output['report'], feature_names[counter], (eval_X.shape[1]-t)) # get metrics
        df.loc[len(df.index)]= datarow # add new metrics to dataframe
        print(feature_names[counter], t) # monitor progress 
        counter +=1 # increase counter
    
        eval_X_copy = perm_feat(16,18,t,eval_X) # permute doy
        perm_output = perm_assess(model, eval_X_copy, y) # assess
        datarow = pi_info(perm_output['confusion_matrix'], perm_output['report'], feature_names[counter], (eval_X.shape[1]-t)) # get metrics
        df.loc[len(df.index)]= datarow # add new metrics to dataframe
        print(feature_names[counter], t) # monitor progress
        counter +=1
    return df

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

def best_params(seed, prefix, file = 'vrnn', metric = 'val_f1'):
    """
    Extract the best parameters for the hyperparameter optimization procedures

    Parameters
    ----------
    seed : random number seed used to initialize hyperparameter optimization
    prefix: trial prefix
    file: file prefix
    metric: metric used for hyperparameter optimization

    Returns
    -------
    space_best : parameters for best model
    """
    trials = joblib.load(path+prefix+str(seed)+'.pkl') # load in the trials 
    rnn_df = read_csv(path+file+str(seed)+'.csv', header = 0, index_col = 0) # read in trial outputs in dataframe format
    bid = rnn_df.loc[rnn_df.val_f1 == max(rnn_df[metric])].index.values[0] # get the indices with the maximum metric value
    space_best = trials.results[bid]['params'] # use indices to extract parameters from the trials 
    return space_best # get best set of parameters according to optimization run

def full_lb(model, params, train_X, train_y, test_X, test_y, name, atype = 'VRNN', metric = 'f1_score'):
    """
    Fit the model, evaluate the model and run permutation importance 
    Get the average of those runs

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
    metric: optimization metric

    Returns
    -------
    eval_run : metrics for each iteration
    avg_val: average of the metrics average: val_f1, val_loss, train_f1, train_loss 
    """
    # assign the weights 
    weights = dict(zip([0,1,2,3], [params['weights_0'], params['weights_1'], params['weights_2'], params['weights_3']]))
    # assign the callback and weight type based on the model type
    class_weights = weights # assign class weights as weights
   
    # fit the model 
    history = model.fit(train_X, train_y, 
                        epochs = params['epochs'],
                        batch_size = params['batch_size'],
                        verbose = 2,
                        shuffle=False,
                        class_weight = class_weights)
        
    y_prob = model.predict(test_X) # make prediction
    y_pred = to_label(y_prob) # get the predictions
    y_label = to_label(test_y) # get the observed targets

    # run permutation importance analysis
    rnn_perm = perm_importance(model,confusion_matrix(y_label, y_pred), class_report(y_label,y_pred), test_X, y_label)    
    rnn_perm['metric_diff'] = abs(rnn_perm[metric]-rnn_perm[metric][0]) # get change in model performance after features were permutated
    rnn_perm.sort_values(by = ['metric_diff'], axis=0, ascending=False, inplace = True)
    rannum = random.randrange(1, 100000,1) # add random number as an identifier for each iteration
    rnn_perm['ID'] = rannum
    rnn_perm.to_csv(path+'/perm_results/perm_imp_'+str(rannum)+'.csv') # save singel iteration output
    del history, model

# assign input & output sequence length
n_input = 13 # lookback
n_output = 1 # prediction timesteps

train = read_csv(path+'train_data.csv', header = 0, index_col = 0) # VV only
test = read_csv(path+'test_data.csv', header = 0, index_col = 0) # VV only

# reshape data 
train_X, train_y, train_dft = to_supervised(train, train['TID'],1, n_input, n_output)
test_X, test_y, test_dft = to_supervised(test, test['TID'],n_output, n_input, n_output)

 # extract the indexes with rows that have unknown targets (i.e., values == -1)
 # all unknown values were previously assigned a value of -1 
a = np.where(test_X[:,12,0]==-1) # column 12 pertains to a behavior column
test_X = np.delete(test_X,a[0],axis =0) # delete unknown target rows
test_y = np.delete(test_y,a[0],axis =0) # delete unknown target rows
test_dft = np.delete(test_dft,a[0],axis =0) # delete unknown target rows
del a 

y_label = to_label(test_y) # extract the labels for the test data, to be used later

space_best = best_params(713,'vanilla_rnn_vv_trials_seed', 'vrnn', 'val_f1')
start = time.perf_counter()

# run the permutation importance 150 times
start_time = time.time() # get start time to track computation time
for j in range(150):
    full_lb(hyp_rnn_nest(space_best,26,4), space_best, train_X, train_y, test_X, test_y, 'vrnn_lb13')
    print(j) # track progress 
print('took', (time.time()-start_time)/60, ' minutes') # print the time lapsed 