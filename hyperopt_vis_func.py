# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet
"""
# Load libraries
import gc
import numpy as np
from numpy import random, newaxis, argmax
import math
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import pandas as pd
from pandas import read_csv, DataFrame, concat, Series, merge
import os, time, itertools, random, logging
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error, f1_score, make_scorer, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split,StratifiedKFold, StratifiedShuffleSplit, cross_val_score, RepeatedStratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
#from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from skopt.plots import plot_convergence
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, Conv1D, Activation, RepeatVector, TimeDistributed, Flatten, MaxPooling1D, ConvLSTM2D
#from tensorflow.keras.preprocessing import sequencef
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score, GeometricMean
#from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import py_func
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import keras.backend as K
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials, trials_from_docs
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
from numba import cuda
import joblib
import seaborn as sns
import ray
from operator import itemgetter
import scipy
from scipy import stats
from scipy.stats import uniform, loguniform, randint
import threading
from threading import Thread
import concurrent.futures
from matplotlib.backends.backend_pdf import PdfPages
from functools import reduce
import glob

# Load and Prepare Dataset
# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

#########################
#### Data Formatting ####
#########################
def trials_to_df(trials):
    """
    converts hyperoptimization experiment results to dataframe

    Parameters
    ----------
    trials : hyperopt trials

    Returns
    -------
    df : dataframe of the hyperparameters, loss and metrics for each hyperopt experiment

    """
    results = trials.results # extract list of trial results
    results = pd.DataFrame(results) # convert to dataframe
    params_list = results.params.to_list() # convert all parameters from hyperopt exp into a list
    params_df = pd.DataFrame(params_list) # convert params for each hyperopt exp into dataframe 
    avg_results = results.avg_eval.to_list() # converte all all avg results for each hyperopt experiment into list                
    avg_results = pd.concat(avg_results, axis = 1).transpose() # convert each avg result into dataframe and transpose so each row represents a hyperoptimization experiment
    df = pd.concat([params_df, avg_results], axis = 1) # concatenate experiment parameters and metrics into a single dataframe
    return df

def hyperopt_progress(df, metrics):
    """
    Ordered series of the hyperopt experiments to track the progress of the parameter optimization

    Parameters
    ----------
    df : dataframe, hyperopt trial results.
    metrics : metrics to visualize

    Returns
    -------
    fig : figure of hyperparameter experiment progression

    """
    nrow = math.ceil(len(metrics)/2) # get half of metrics being tracked to assign number of subplot rows
    # specify the subplots
    fig, ax = plt.subplots(nrows = nrow,
                              ncols = 2,
                              sharex = True, 
                              sharey = False,
                              figsize = (10,nrow)) 
    counter = 1 # start counter
    # for each metric
    for metric in metrics:
        plt.subplot(nrow,2,counter) # use counter as the subplot index
        plt.plot(df.index, df[metric], label = metric) # plot the metric by the index
        plt.ylabel(metric) # label the metric 
        counter +=1 # increase the counter
    fig.supxlabel('runs') # add x axis label
    fig.tight_layout() 
    return fig

    return(fig)

def train_val_comp(df):
    """
    Plot the training and validation loss and metrics against each other

    Parameters
    ----------
    df : dataframe, hyperopt trial results

    Returns
    -------
    fig: figure, plots of training v. validation loss, f1 and accuracy

    """
    fig, axis = plt.subplots(1,3,figsize=(10,3))
    plt.subplot(1, 3, 1) # divide the plot space 
    # plot the relationship between val loss and train loss
    sns.regplot(x = df['val_loss'], 
                y = df['train_loss'], 
                fit_reg = True,
                color = '#377eb8') 
    # plot the relationship between val loss and train loss
    plt.subplot(1, 3, 2) # divide the plot space 
    sns.regplot(x = df['val_f1'], 
                y = df['train_f1'], 
                fit_reg = True,
                color = '#ff7f00') 
    plt.subplot(1, 3, 3) # divide the plot space 
    # plot the relationship between val loss and train loss
    sns.regplot(x = df['val_acc'], 
                y = df['train_acc'], 
                fit_reg = True,
                color = '#4daf4a') 
    fig.tight_layout()
    return fig

def kdeplots(df,metrics):
    """
    Generate kernal density plots of the metrics of interest against the validation loss

    Parameters
    ----------
    df : dataframe, hyperopt trial results.
    metrics : metrics to visualize

    Returns
    -------
    fig : figure of hyperparameter experiment progression

    """
    nrow = math.ceil(len(metrics)/3) # get half of metrics being tracked to assign number of subplot rows
    # specify the subplots
    fig,ax = plt.subplots(nrow,3, figsize = (10,6))
    counter = 1 # start counter 
    # for each metric
    for metric in metrics:
        plt.subplot(nrow,3,counter) # use counter as the subplot index
        # plot kernel density plot
        sns.kdeplot(x = df[metric], 
                    y = df['val_loss'], 
                    shade = True, 
                    thresh = 0.05, 
                    legend = True, 
                    color = 'gray')
        plt.xlabel(metric)  # label the metric 
        counter+=1 # increase the counter
    fig.tight_layout()  
    return fig
    
def trial_correg_plots(trials, params, monitor = ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc']):
    """
    Wrapper to format hyperopt trials into dataframes and 
    
    Parameters
    ----------
    trials : hyperopt trials 
    params : list,
        Hyperparameter/parameter metrics. The default is ['val_f1'].
    monitor : list, optional
        Loss and performance metrics. The default is ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc'].
    filename: str, optional
        Name to save figures. The default is None.
        
    Returns
    -------
    df : dataframe
        dataframe of the trial results.

    """
    
    df = trials_to_df(trials) # convert hyperopt experiment results to a dataframe
    tv_fig = train_val_comp(df) # generate plot to compare training v. validation loss and performance
    hp_fig = hyperopt_progress(df, monitor + params) # visualize hyperopt experiment progression
    kd_fig = kdeplots(df, params) # kernal density plot of hyperparameters against validation loss
    return df

def trial_correg_pdf(path, filename, params, monitor = ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc']):
    """
    Function to format hyperopt trials into dataframes and visualize results
    
    Parameters
    ----------
    path : str, file direction of hyperopt output 
    filename: str, 
            filename
    params : list,
        Hyperparameter/parameter metrics
    monitor : list, optional
        Loss and performance metrics. The default is ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc'].
    
    Returns : 
    -------
    df: dataframe of the trial results.

    """
    trials = joblib.load(path + filename + '.pkl') # load hyperopt trial file
    df = trials_to_df(trials) # convert hyperopt experiment results to a dataframe
    df.to_csv(path+filename+'_results.csv')
    tv_fig = train_val_comp(df) # generate plot to compare training v. validation loss and performance
    hp_fig = hyperopt_progress(df, monitor + params) # visualize hyperopt experiment progression
    kd_fig = kdeplots(df, params) # kernal density plot of hyperparameters against validation loss
    # create PDF for results and save figures in the pdf
    with PdfPages(path+filename+'_results.pdf') as pdf:
        pdf.savefig(tv_fig)
        pdf.savefig(hp_fig)
        pdf.savefig(kd_fig)
        plt.close()
        plt.figure(figsize=(10, 10)) # assign figure size
        # calculate the correlation between hyperparameters and metrics
        vcorr = df.loc[:, monitor + params].corr() 
        # plot heatmap of correlations
        sns.heatmap(vcorr, 
                    xticklabels=vcorr.columns,
                    yticklabels=vcorr.columns,
                    cmap = 'vlag') 
        pdf.savefig()
    return df

# filelist = []
# for file in glob.glob('ende_f1*.pkl'):
#     file = file[:-4]
#     filelist.append(file)

# metrics = ['val_loss','train_loss', 'drate','weights_0','weights_2','weights_3','epochs','lookback','neurons_n0','neurons_n1',
#            'hidden_layers','hidden_n0','td_neurons',]

# filelist = []
# for file in glob.glob('vrnn_gru_full*.pkl'):
#     file = joblib.load(file)
#  #   file = file[:-4]
#     filelist.append(file)
    
# metrics = ['val_loss', 'train_loss', 'drate','weights_0','weights_2',
#            'weights_3','val_f1','epochs','lookback','neurons_n','hidden_layers','hidden_n0',]

def get_ci(ary, threshold):
    """
    extract credible interval limits for each metric in the array

    Parameters
    ----------
    ary : array, metric values 
    threshold : float, 0-1
        the credible interval threshold (alpha)

    Returns
    -------
    lp : array, lower credible intervals
    up: array, upper credible intervals

    """
    lci = (1-threshold)/2 # get lower credible interval threshold
    uci = 1- lci # get upper credible interval threshold
    # create empty list to population metrics 
    lp = [] 
    up = []
    llim = int(ary.shape[0]*lci)
    ulim = int(ary.shape[0]*uci)
    for i in range(ary.shape[1]): # for each metric
        svalues = np.sort(ary[:,i]) # sort the values of metrics
        lp.append(svalues[llim]) # get the value at the lower credible interval
        up.append(svalues[ulim]) # get the value at the upper credible interval
    # transform list into array
    lp = np.array(lp) 
    up = np.array(up)
    return lp, up # return the arrays of credible intervals for the metrics

def hypoutput(path, modelname, params, ci = 0.90, burnin=200, maxval=1000):
    '''
    combine the hyperopt values and get nonparametric median and 95% intervals

    Parameters
    ----------
    path : str,
        location of hyperopt files
    modelname : str, 
        model filename prefix
    params : list, 
        parameters/hyperparameters to evaluate
    ci: numeric, 
        credible interval alpha threshold. Default is 0.90.
    burnin : int, 
        number of initial experiments to discard. Dafault is 200.
    maxval : int, 
        maximum number of experiments to compare. Default is 1000.

    Returns
    -------
    output : dataframe, 
        formatted minimum loss, median performance metrics and median hyperparameter values with credible intervals

    '''
    loss = ['train_loss','val_loss']
    metrics = ['train_f1','val_f1','train_acc','val_acc'] + params
    dflist = [] # empty list for dataframes to combine of hyperopt outputs
    for file in glob.glob(path + modelname + '*_results.csv'): # load all the hyperopt trials using the list of filenames
        trial_df =  read_csv(file, header =0, index_col = 0) # transform each file into a dataframe
        trial_df = trial_df.iloc[burnin:maxval,:] # subset experiments based on burn-in and max value
        dflist.append(trial_df) # add to the list of dataframes
    trial_df = pd.concat(dflist) # concatenate all dataframes
    trial_df = trial_df[loss + metrics] # subset dataframe by metrics
    lci, uci = get_ci(trial_df.to_numpy(), ci) # get credible intervals 
    lci = np.round(lci,2) # round lower interval
    uci = np.round(uci,2) # round upper interval
    medval = trial_df[metrics].median(axis = 0) # get median for hyperparameters
    medval = medval.round(2) 
    minval = trial_df[loss].min(axis = 0) # get minimum loss
    minval = minval.round(2)
    d1 = [str(x) + ' (' +str(y) +','+ str(z) + ')' for x, y, z in zip(minval, lci[:2], uci[:2])] # format loss values
    d2 = [str(x) + ' (' +str(y) +','+ str(z) + ')' for x, y, z in zip(medval[:4], lci[2:6], uci[2:6])] # format float values
    d3 = [str(x) + ' (' +str(y) +','+ str(z) + ')' for x, y, z in zip(medval[4:].astype('int32'), lci[6:].astype('int32'), uci[6:].astype('int32'))] # format integer values
    output = [modelname] + d1 + d2 + d3 # add together into same list
    # convert to dataframe
    output = pd.DataFrame(output, 
                          index = ['model'] + 
                          loss + metrics, columns= ['summary']) 
    return output # output values and intervals
   
# modelnames= ['vrnn_f1_GRU_behavior', 'vrnn_f1_GRU_full','vrnn_f1_GRU_extrinsic',
#             'vrnn_f1_LSTM_behavior', 'vrnn_f1_LSTM_full','vrnn_f1_LSTM_extrinsic',
#             'ende_f1_GRU_behavior', 'ende_f1_GRU_full','ende_f1_GRU_extrinsic',
#             'ende_f1_LSTM_behavior', 'ende_f1_LSTM_full','ende_f1_LSTM_extrinsic',
#             ]

# modelnames= ['vrnn_GRU_full', 'vrnn_LSTM_full','ende_GRU_full','ende_LSTM_full']


# ende_df = pd.DataFrame(columns = ['model','val_loss','train_loss', 'drate','weights_0','weights_2','weights_3','epochs','lookback','neurons_n0','neurons_n1',
#            'hidden_layers','hidden_n0','td_neurons'])
# vrnn_df = pd.DataFrame(columns = ['model','val_loss', 'train_loss', 'drate','weights_0','weights_2',
#            'weights_3','val_f1','epochs','lookback','neurons_n','hidden_layers','hidden_n0'])

# # run the hypoutput for all model types
# for mn in modelnames:
#     if mn[0:4] == 'vrnn':
#         metrics = ['val_loss', 'train_loss', 'drate','weights_0','weights_2',
#            'weights_3','val_f1','epochs','lookback','neurons_n','hidden_layers','hidden_n0']
#         entry = hypoutput(mn, metrics)
#         vrnn_df.loc[len(vrnn_df.index)] = entry 
#     else:
#         metrics = ['val_loss','train_loss', 'drate','weights_0','weights_2','weights_3','epochs','lookback','neurons_n0','neurons_n1',
#            'hidden_layers','hidden_n0','td_neurons']
#         entry = hypoutput(mn, metrics)
#         ende_df.loc[len(ende_df.index)] = entry 
        
# df = pd.concat([vrnn_df,ende_df], axis = 0, join ='outer',ignore_index = True)
# df.to_csv('behavior_f1_loss_update_sumcombo90.csv')

# dflist = [] # empty list for dataframes to combine of hyperopt outputs
# for file in glob.glob('vrnn_f1_GRU_full*.pkl'): # get all hyperopt files with prefix
#     trials = joblib.load(file) # load each file
#     trial_df = trial_to_df(trials) # trasnform into dataframe
#     dflist.append(trial_df) # add to list
# trial_df = pd.concat(dflist,ignore_index = True) # concatenate all dataframes      

# trial_df.to_csv('vrnn_gru_full_hyperopt_combined.csv')
# kdeplots(trial_df, ['val_f1','drate','weights_0','weights_2',
#            'weights_3','epochs','lookback','neurons_n','hidden_layers','hidden_n0'])

# trials = joblib.load('vrnn_GRU_full_75879.pkl')

# for file in filelist:
#     trial_df = trial_correg_pdf(path, file,['status','params','loss'], metrics = metrics)
#     print(file)

# trial_df = trial_correg_pdf(path, 'vrnn_GRU_full_42577',['status','params','loss'], metrics = metrics)

# trial0 = joblib.load(filelist[0]+'.pkl')
# trial1 = joblib.load(filelist[1]+'.pkl')
# trial2 = joblib.load(filelist[2]+'.pkl')
# trial_df0 = trial_to_df(trial0, drop_col = ['status','params','loss'])
# trial_df1 = trial_to_df(trial1, drop_col = ['status','params','loss'])
# trial_df2 = trial_to_df(trial2, drop_col = ['status','params','loss'])

# trial_df = trial_df0.append(trial_df1, ignore_index = True)
# trial_df = trial_df.append(trial_df2, ignore_index = True)
# for file in filelist:
#     trials = joblib.load(file+'.pkl')
#     trial_df = trial_to_df(trials, drop_col = ['status','params','loss'])
    
#     print(file)
    
#     trial_to_df(trials, drop_col = ['status','params','loss'])


# trial_df = trial_correg_pdf(path, 'vrnn_LSTM_full_92997',['status','params','loss'], metrics = metrics)


# ende_df = pd.DataFrame(columns = ['model'] + metrics)
# vrnn_df = pd.DataFrame(columns = ['model'] + metrics)

# for file in filelist:
#     trials = joblib.load(file + '.pkl')
#     trial_df = trial_to_df(trials, ['status','params','loss'])  
#     trial_np = trial_df[metrics].to_numpy()
#     lci, uci = get_ci(trial_np)
#     medval = np.median(trial_np,axis = 0)
#     minval = np.min(trial_np,axis = 0)
#     lci[0:6] = np.round(lci[0:6],2)
#     uci[0:6] = np.round(uci[0:6],2)
#     minval[0:2] = np.round(minval[0:2],2)
#     medval[2:6] = np.round(medval[2:6],2)
    
#     d1 = [str(x) + ' (' +str(y) +','+ str(z) + ')' for x, y, z in zip(minval[:2], lci[:2], uci[:2])]
#     d2 = [str(x) + ' (' +str(y) +','+ str(z) + ')' for x, y, z in zip(medval[2:6], lci[2:6], uci[2:6])]
#     d3 = [str(x) + ' (' +str(y) +','+ str(z) + ')' for x, y, z in zip(medval[6:].astype('int32'), lci[6:].astype('int32'), uci[6:].astype('int32'))]
    
#     entry = [file] + d1 + d2 + d3
#     vrnn_df.loc[len(vrnn_df.index)] = entry 
#    # ende_df.loc[len(ende_df.index)] = entry
    
# vrnn_df.to_csv('vrnn_behavior_update.csv')
# ende_df.to_csv('ende_behavior_update.csv')

# trial_df = trial_correg_plots(trials, ['status','params','loss'], metrics = metrics)     

# get_ci_df(trial_df[trial_df['hidden_layers'] ==1],0.5)

def sum_function(df,filename):
    grand_mean = df.mean(axis = 0) # calcualte grand median
    grand_median = df.median(axis = 0) # calcualte grand median
    grand_sd = df.std(axis = 0) # calculate grand standard deviateion
    grand_mad = df.mad(axis = 0) # calculate grand standard deviateion
    l50, u50 = get_ci_df(df, 0.50) # get 50% credible intervals 
    l80, u80 = get_ci_df(df, 0.80) # get 80% credible intervals
    l90, u90 = get_ci_df(df, 0.90) # get 90% credible intervals
    l95, u95 = get_ci_df(df, 0.95) # get 95% credible intervals
    summary_df = pd.concat([grand_mean, grand_sd, grand_median, grand_mad,l95, u95, l90, u90, l80,u80,l50,u50],axis = 1) # generate summary table 
    summary_df.columns = ['mean','sd','median', 'mad','lci95','uci95','lci90','uci90','lci80','uci80','lci50','uci50'] # add column names 
    summary_df['model'] = filename
    summary_df.to_csv(filename+'_summary.csv') #save output
    return summary_df

#markov_df = read_csv('model_comparison_behavior21_markov.csv', header = 0, index_col = 0)
#actdist_df = read_csv('model_comparison_behavior21_actdist.csv', header = 0, index_col = 0)

# a = sum_function(markov_df.iloc[:,2:], 'markov21_behavior_compcheck')
# b = sum_function(actdist_df.iloc[:,2:], 'actdist21_behavior_compcheck')
# df_tabs = [markov_df, actdist_df]
# df_tabs = [act_dist21, act_dist23, markov21, markov23, rnn_median21, rnn_median21_binom, rnn_best23, rnn_best23_binom]
# filename = ['markov21_check','act_dist21_check']

# filename = ['act_dist21', 'act_dist23', 'markov21', 'markov23', 'rnn_median21', 'rnn_median21_binom', 'rnn_best23', 'rnn_best23_binom']
# empty_list = []
# for i in range(len(df_tabs)):
#     a = sum_function(df_tabs[i].iloc[:,2:], filename[i])
#     empty_list.append(a)

# cross_comp_tab = pd.concat(empty_list)    
# cross_comp_tab.to_csv('cross_comparison_behavior_ci_markov_act_21.csv')
# trials = []
# for file in glob.glob('ende_GRU_extrinsic_56924' +'*.pkl'): # for files with this prefix
#     print(file)    
#     file = joblib.load(file) # load each file 
#     trial_df = trial_to_df(file)  # convert to dataframe
#     print(trial_df.shape)
#     trials.append(trial_df)
    
def get_ci_df(ary,percent):
    lp = []
    up = []
    lb = (1-percent)/2
    ub = 1-((1-percent)/2)
    llim = int(ary.shape[0]*lb)-1
    ulim = int(ary.shape[0]*ub)-1
    
    collist = list(ary.columns) 
    for i in collist:
        svalues = ary.sort_values(i, ignore_index=True)
        lp.append(svalues[i][llim])
        up.append(svalues[i][ulim])
    lp = pd.Series(lp, index = collist)
    up = pd.Series(up, index = collist)
    return lp, up


# calculate rhat to check for convergence
def convergence_sum(prefix, metrics, burnin = 0, maxval = 1000):
    trials = []
    var_list = []
    mean_list = []
    nhidden = 0
    for file in glob.glob(prefix +'*.pkl'): # for files with this prefix
        file = joblib.load(file) # load each file 
        trial_df = trial_to_df(file)  # convert to dataframe
        trial_df = trial_df[metrics] # subset metrics of interest only
        trial_df = trial_df.iloc[burnin:maxval,] # drop burnin draws
        trial_df['hidden_n0'] = np.where(trial_df['hidden_layers'] == 0, np.NaN, trial_df['hidden_n0'])
        nhidden += trial_df[trial_df['hidden_layers'] == 1].shape[0]
        trials.append(trial_df) # add to trial list
        trial_var = trial_df.var(axis = 0) # get trial variance 
        trial_mean = trial_df.mean(axis = 0) # get trial variance 
        var_list.append(trial_var)  # add to variance list
        mean_list.append(trial_mean)  # add to variance list        
    n = trial_df.shape[0] # get number of draws 
    nhidden = round(nhidden/3,0) # get estimate of n, making assumption that get equal number of hidden layers choosen in each chain, not true though
    j = len(var_list)
    within_var = pd.concat(var_list,axis = 1) # merge trial variances into single dataframe
    w = within_var.mean(axis=1) # get within variance
    within_mean = pd.concat(mean_list,axis = 1)
    df = pd.concat(trials,axis = 0, ignore_index=True) # merge all datapoints into same dataframe
    
    grand_mean = df.mean(axis = 0) # get grand mean
    # calculate the between variance
    bvalues = within_mean.subtract(grand_mean, axis = 0) # get difference between trial mean and grand mean
    bvalues = bvalues**2
    bvalues = bvalues.sum(axis=1) # sum them together
    b = n/(j-1)*bvalues # square and multiply by data factor
    n = trial_df.shape[0] # get number of draws 
    rhat = (((n-1)/n)*w + (1/n)*b)/w # calculate rhat
    grand_median = df.median(axis = 0) # calcualte grand median
    grand_sd = df.std(axis = 0) # calculate grand standard deviateion
    grand_mad = df.mad(axis = 0) # calculate grand standard deviateion
    l50, u50 = get_ci_df(df, 0.50) # get 50% credible intervals 
    l80, u80 = get_ci_df(df, 0.80) # get 80% credible intervals
    l90, u90 = get_ci_df(df, 0.90) # get 90% credible intervals
    l95, u95 = get_ci_df(df, 0.95) # get 95% credible intervals
    summary_df = pd.concat([grand_mean, grand_sd, grand_median, grand_mad, rhat, l95, u95, l90, u90, l80,u80,l50,u50],axis = 1) # generate summary table 
    summary_df.columns = ['mean','sd','median', 'mad','rhat','lci95','uci95','lci90','uci90','lci80','uci80','lci50','uci50'] # add column names 
    summary_df.to_csv(prefix + '_'+str(burnin)+'_summary.csv') #save output
    df['trial'] = np.repeat(range(0,j),n)
    return trials, df, summary_df # return summary table 

# metrics_vrnn = ['drate','weights_0','weights_2',
#            'weights_3','lookback','epochs','neurons_n','hidden_layers','hidden_n0']

# metrics_ende = ['drate','weights_0','weights_2','weights_3','lookback','epochs','neurons_n0','neurons_n1',
#            'hidden_layers','hidden_n0','td_neurons']

# prefix = 'vrnn_f1_GRU_extrinsic'
# burnin = 200


def chain_plots(trials, metric, ax = None):
    colors = ['coral','cyan','goldenrod']
    for i in range(len(trials)):    
        sns.lineplot(ax = ax, data = trials[i],x = trials[i].index, y = metric, alpha =0.5, color = colors[i])
        ax.set_xlabel('drate')
    return ax

def hist_plots(trials, metric,ax = None):
    colors = ['coral','cyan','goldenrod']
    for i in range(len(trials)):   
        sns.histplot(ax = ax, data = trials[i],x = metric, bins = 20, alpha =0.5, color = colors[i])
    return ax

def kde_plots(trials, metric, ax= None):
    colors = ['coral','cyan','goldenrod']
    for i in range(len(trials)):  
        sns.kdeplot(ax =ax, data = trials[i], x = metric, y = 'val_loss', shade = True, legend = False, color = colors[i], alpha =0.5)
    return ax
    
def loss_plots(trials):
    #n = int(combo_df.shape[0]/3)
    fig, ax = plt.subplots(2,2,gridspec_kw={'width_ratios': [2, 1]}, figsize = [10,3])
    chain_plots(trials, 'val_loss', ax[0,0])
    hist_plots(trials, 'val_loss', ax[0,1])
    chain_plots(trials, 'train_loss', ax[1,0])
    hist_plots(trials, 'train_loss', ax[1,1])
    fig.tight_layout()
    return fig

def parameter_plots(trials, metrics):
    fig, ax = plt.subplots(len(metrics),3, gridspec_kw={'width_ratios': [2, 1, 1]},figsize = (10,12))
    for i in range(len(metrics)):
        chain_plots(trials, metrics[i], ax[i,0])
        hist_plots(trials, metrics[i], ax[i,1])
        kde_plots(trials, metrics[i], ax[i,2])
    fig.tight_layout()
    return fig

def trial_chains_output(prefix, metrics, burnin = 0):
    trials, combo_df, summary_df = convergence_sum(prefix, ['val_loss','train_loss']+metrics, burnin = burnin, maxval = 2000)
    summary_df.loc[['val_loss','train_loss','drate'],:] = round(summary_df.loc[['val_loss','train_loss','drate'],:],2)
    summary_df.loc[:,['mean','sd','median','mad','rhat']] = round(summary_df.loc[:,['mean','sd','median','mad','rhat']],2)
    sum_tab = summary_df[['mean','sd','median','mad','lci95','uci95','rhat']]
    ci_tab = summary_df[['median','mad','lci95','uci95','lci90','uci90','lci80','uci80','lci50','uci50']]
   # trainval_plot = sns.regplot(combo_df['train_loss'], combo_df['val_loss'], fit_reg = True,color = 'orange') # plot the relationship between val loss and train loss
  #  lossplots = loss_plots(trials)
  #  paramplots = parameter_plots(trials,metrics)
    
    with PdfPages(path+prefix+'_multichain_results.pdf') as pdf:       
        lossplots = loss_plots(trials)
        pdf.savefig(lossplots) # save figure
        plt.close() # close page
        
        paramplots = parameter_plots(trials,metrics)
        pdf.savefig(paramplots) # save figure
        plt.close() # close page
        
        # generate subplots to organize table data 
        fig, ax = plt.subplots(3,1, sharex = False, sharey = False, figsize=(10,12))
        sns.regplot(ax =ax[0], data = combo_df, x = 'train_loss', y = 'val_loss', fit_reg = True,color = 'orange')
  
        ax[1].table(cellText=sum_tab.values,colLabels = sum_tab.columns, rowLabels=sum_tab.index,loc='center')
        ax[1].axis('tight') 
        ax[1].axis('off')
        
        ax[2].table(cellText=ci_tab.values,colLabels = ci_tab.columns, rowLabels=ci_tab.index,loc='center')
        ax[2].axis('tight') 
        ax[2].axis('off')
        pdf.savefig() # save figure
        plt.close() # close page
        
# trial_chains_output('ende_f1_LSTM_full', metrics_ende, 200)
# trial_chains_output('vrnn_f1_GRU_behavior', metrics_vrnn, 200)      
        
# modelnames= ['vrnn_f1_GRU_behavior', 'vrnn_f1_GRU_full','vrnn_f1_GRU_extrinsic',
#             'vrnn_f1_LSTM_behavior', 'vrnn_f1_LSTM_full','vrnn_f1_LSTM_extrinsic',
#             'ende_f1_GRU_behavior', 'ende_f1_GRU_full','ende_f1_GRU_extrinsic',
#             'ende_f1_LSTM_behavior', 'ende_f1_LSTM_full','ende_f1_LSTM_extrinsic',
#             ]

# for mn in modelnames:
#     if mn[:4] == 'vrnn':
#         metrics = ['drate','weights_0','weights_2',
#            'weights_3','lookback','epochs','neurons_n','hidden_layers','hidden_n0']
#     else:
#         metrics = ['drate','weights_0','weights_2','weights_3','lookback','epochs','neurons_n0','neurons_n1',
#            'hidden_layers','hidden_n0','td_neurons']
#     trial_chains_output(mn, metrics, 200)

# modelnames= ['vrnn_f1_GRU_behavior', 'vrnn_f1_GRU_full','vrnn_f1_GRU_extrinsic',
#             'vrnn_f1_LSTM_behavior', 'vrnn_f1_LSTM_full','vrnn_f1_LSTM_extrinsic',
#             'ende_f1_GRU_behavior', 'ende_f1_GRU_full','ende_f1_GRU_extrinsic',
#             'ende_f1_LSTM_behavior', 'ende_f1_LSTM_full','ende_f1_LSTM_extrinsic',
#             ]

# trials, trial_df, summary_df = convergence_sum('vrnn_GRU_behavior', ['val_loss','train_loss','drate','weights_0','weights_2',
#            'weights_3','lookback','neurons_n','hidden_layers','hidden_n0','epochs'],200,1000)

# comp_df = pd.DataFrame(columns = list(trial_df.columns))
# for file in filelist:
#     trial_df = trial_to_df(file)   
#     comp_df.loc[len(comp_df.index)] = trial_df.iloc[999,:]
#     #trial_df.sort_values('val_loss', ascending=False,inplace =True)
#     bid = trial_df.nsmallest(1,'val_loss').index.values[0]
#     comp_df.loc[len(comp_df.index)] = trial_df.iloc[bid,:]
    
# metrics = ['drate','weights_0','weights_2',
#            'weights_3','lookback','neurons_n','hidden_layers','hidden_n0']

# comp_df = pd.DataFrame(columns = list(trial_df.columns))
# for file in filelist:
#     trial_df = trial_to_df(file)   
#  #   comp_df.loc[len(comp_df.index)] = trial_df.iloc[999,:]
#     #trial_df.sort_values('val_loss', ascending=False,inplace =True)
#     bid = trial_df.nsmallest(1,'val_loss').index.values[0]
#     comp_df.loc[len(comp_df.index)] = trial_df.iloc[bid,:]
    
# comp_df.to_csv('compare_vrrn_gru_full.csv')
# trial_df.sort_values('val_loss', ascending=False,inplace =True)
# bid = trial_df.nsmallest(1,'val_loss').index.values[0]
# comp_df.loc[len(comp_df.index)] = trial_df.iloc[bid,:]
# params = trials.results[bid]['params']
    

# subdf = trial_df[trial_df['val_loss'] < 0.44]
# best_results = trials.results[bid]['results']
# best_results =round(best_results,3)
# best_sum = trials.results[bid]['summary']
# best_results.loc[len(best_results.index)]= round(best_sum['mean'],3).map(str)+ ' (' + round(best_sum['std'],3).map(str) + ')'
# best_results.to_csv('leprosula_anomaly_rain_solar_temp_best_405.csv')

# bids = trial_df.nsmallest(10,'val_loss')
# bids.loc[:,('lag','lookback')]


# unique_trials = trial_df.drop_duplicates(subset = ['lag','lookback'])
# ubids = unique_trials.nlargest(10,'val_roc')
# ubids = round(ubids.iloc[:,np.r_[11:13,1,3,5,7,9,0,2,4,6,8]],3)
# ubids = round(ubids.iloc[:,np.r_[15,17,1,3,5,7,9,0,2,4,6,8]],3)

# seed(1)
# tf.random.set_seed(2)
# SEED = 405 #used to help randomly select the data points
# DATA_SPLIT_PCT = 0.2
# rcParams['figure.figsize'] = 8, 6
# LABELS = ["no fruit","fruit"]

# # import climate data
# kian_era =  read_csv('kian_era_std.csv', header = 0)
# rano_era =  read_csv('rano_era_std.csv', header = 0)

# kian_era.columns = ["ID", "date.time", "ctime", "date", "time", "temp2m.max", "temp2m.min", 
#                     "temp2m.mean", "rain", "relhum.max", "relhum.min", "relhum.mean", "cloud.cover", 
#                     "solar.rad", "uv.rad", "temp.max", "temp.min", "temp.mean", "soiltemp.max", 
#                     "soiltemp.min", "soiltemp.mean", "soilmoist.max", "soilmoist.min", "soilmoist.mean"]
# rano_era.columns = kian_era.columns

# # subset relevant clim columns
# kian_clim = kian_era.copy()
# kian_clim = kian_clim.iloc[:,[4,7,8,11,12,13,17,20,23]]

# rano_clim = rano_era.copy()
# rano_clim = rano_clim.iloc[:,[4,7,8,11,12,13,17,20,23]]

# # import training datafiles
# dataset =  read_csv('rahiaka_formatted.csv', header = 0, index_col = 0) # VV only
# datasub = dataset.copy() # make a copy to manipulate 
# datasub = datasub.iloc[:, [0,1,4,5,6,7,10,11,12]] # get rid of these covariates

# datasub = datasub[datasub['fruit'] != -1]

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

# train_val, test = split_dataset(datasub,2015)
# train, valid = split_dataset(train_val,2013)

def to_supervised(data, rano_clim, kian_clim, lookback, lag = 0, yval = 'fruit'):
    """
    Format training data for multivariate, multistep time series prediction models

    Parameters
    ----------
    data : phenology data
    DSID : dataset identifier
    rnp_clim: ranomafana climate data
    kian_clim: kianjavato climate data
    lookback : lookback period
    lag: lag timesteps (e.g., lag = -2, lookback = 2 then extract features from t-2 to t-6)
    Returns
    -------
    X : features data in array format (input)
    y : target data in array format (output)
    dft: deterministic features for prediction timesteps in array formate

    """
    # get indice of the time column 
    kt = kian_clim.columns.get_loc('time')
    rt = rano_clim.columns.get_loc('time')
    dt = data.columns.get_loc('time')
    if yval == 'fruit':
        dy = data.columns.get_loc('fruit')
    else:
        dy =data.columns.get_loc('flower')
    
    kian = np.array(kian_clim) # convert climate data into numpy array
    rano = np.array(rano_clim) # convert climate data into numpy array
    data = np.array(data) # convert data into numpy array
    
    X, y, dft = list(), list(), list() # get empty list for X (all features), y (targets), dft (deterministic features in prediction time steps)
    in_start = 0 # set start index as 0
	# step over the entire dataset one time step at a time
    if lag > 0:
        print('lag must be nonpositive')
        exit()
    for _ in range(len(data)):
        ct = data[_,dt] # get time 
        region = data[_,rt] # get region
        if region == 0:
            clim = kian
        else:
            clim = rano
        in_end = int(np.where(clim[:,kt] == ct+1+lag)[0]) # define the end of the input sequence
        in_start = in_end - lookback # define the end of the input sequence
        if in_start >=0:
            X.append(clim[in_start:in_end, 1:5]) # append input sequence to features list
            y.append(data[_,dy]) # append output to the targets list
            dft.append(np.delete(data[_,:], dy)) # append the deterministic features for current timestep to the deterministic features list
    X = np.array(X) # convert list to array
    y = np.array(y, dtype = 'int64') # convert list to array
    dft = np.array(dft) # convert list to array
    return X, y, dft 

def flatten(X):
    '''
    Flatten a 3D array.
    
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

def scale(X, scaler):
    '''
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
    
    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
        
    return X

def plot_monitor(results):
    keys = list(results.history)
    nk = int(len(keys)/2)
    fig, axis = plt.subplots(1,nk,figsize=(5*nk,4))
    for i in range(nk):
        plt.subplot(1, nk, i+1) # divide the plot space 
        plt.plot(results.history[keys[i]], label='train')
        plt.plot(results.history['val_'+keys[i]], label='validation')
        plt.legend()
        plt.title(keys[i])

# Model Evaluation         
def monitoring_plots(result):
    """
    plot the training and validation loss, accuracy, pr_auc, roc_auc

    Parameters
    ----------
    result : history from the fitted model 

    Returns
    -------
    None.
    """
    # plot the loss
    fig, axis = plt.subplots(1,4,figsize=(20,4))
    plt.subplot(1, 4, 1) # divide the plot space 
    try:
        plt.plot(result.history['loss'], label='train')
        plt.plot(result.history['val_loss'], label='validation')
        plt.legend()
        plt.title('loss')
     #   plt.show()
    except:
        print('loss not monitored')
        
    # plot the accuracy
    plt.subplot(1, 4, 2) # divide the plot space 
    try:
        plt.plot(result.history['accuracy'], label='train')
        plt.plot(result.history['val_accuracy'], label='validation')
        plt.legend()
        plt.title('accuracy')
   #     plt.show()
    except:
        print('accuracy not monitored')
    
    # plot the pr_auc
    plt.subplot(1, 4, 3) # divide the plot space 
    try:  
        plt.plot(result.history['pr_auc'], label='train')
        plt.plot(result.history['val_pr_auc'], label='validation')
        plt.legend()
        plt.title('pr_auc')
  #      plt.show()
    except:
        print('pr_auc not monitored')
        
    # plot the roc_auc
    plt.subplot(1, 4, 4) # divide the plot space 
    try:
        plt.plot(result.history['roc_auc'], label='train')
        plt.plot(result.history['val_roc_auc'], label='validation')
        plt.legend()
        plt.title('roc_auc')
        plt.show()        
    except:
        print('roc_auc not monitored')
   
def confusion_mat(y, y_pred, LABELS, normalize = 'true'):
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
    cm = confusion_matrix(y,y_pred, normalize = normalize) # generalize the normalized confusion matrix
    if normalize == None:
        sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt ='d')
    else:
        sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True)
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

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
    reports : list of classification reports to average over

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





def kde_comp_mm(sub_df, groups, metric = 'val_acc', title = None):
    """
    Get kernal density plots for each hyperparameter

    Parameters
    ----------
    groups : the column indices for the hyperparameters of interest
    sub_df : dictionary for the 3 trials being compared. The dictionary should contain the trials in which val_f1 >= 0.4
    ymax : maximum y limit for the graph

    Returns
    -------
    Outputs the kernel density graphs

    """
    ng = len(groups)
    nm = len(metric)
    fig, axis = plt.subplots(ng,nm,figsize=(ng*4,nm*4)) # set the plot space dimensions
    i = 1 # start counter
    for group in groups:
        for m in metric:
            plt.subplot(ng, nm, i) # divide the plot space 
            if sub_df.iloc[:,group].dtype == 'O':
                values = pd.factorize(sub_df.iloc[:,group].values, na_sentinel = None, sort = True)
                g = sns.kdeplot(values[0],sub_df[m], shade = True, 
                                shade_lowest=False, legend = True, color = 'purple')
                plt.xlabel(str(values[1]))
            else:
                g = sns.kdeplot(sub_df.iloc[:,group],sub_df[m], 
                                shade = True, shade_lowest=False, legend = True, color = 'purple') # plot values that don't have nulls for the hyperparameter of interest against the val f1
            i +=1
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

# dflist[0].nsmallest(1,'val_loss').transpose()

def triplot3d(df):
    metrics = ['val_loss','val_acc','val_roc','val_pr']
    df = df.drop_duplicates(subset = ['lag','lookback'])
    fig = plt.figure(figsize = (16,4))
    counter = 1
    for metric in metrics:
        if metric == 'val_loss':
            sub_df = df.nsmallest(n=100, columns = [metric])
        else: 
            sub_df = df.nlargest(n=100, columns = [metric])
        ax = fig.add_subplot(1,4,counter, projection = '3d')
        zdata = sub_df[metric]
        xdata = sub_df['lag']
        ydata = sub_df['lookback']
        ax.plot_trisurf(xdata, ydata, zdata, cmap = 'Greens',edgecolor ='none')
        ax.set_xlabel('lag')
        ax.set_ylabel('lookback')
        ax.set_zlabel(metric)
        counter+=1
    plt.show()
    
def scatplot3d(df):
    metrics = ['val_loss','val_acc','val_roc','val_pr']
    df = df.drop_duplicates(subset = ['lag','lookback'])
    fig = plt.figure(figsize = (16,4))
    counter = 1
    for metric in metrics:
        if metric == 'val_loss':
            sub_df = df.nsmallest(n=100, columns = [metric])
        else: 
            sub_df = df.nlargest(n=100, columns = [metric])
        ax = fig.add_subplot(1,4,counter, projection = '3d')
        zdata = sub_df[metric]
        xdata = sub_df['lag']
        ydata = sub_df['lookback']
        ax.scatter3D(xdata, ydata, zdata,color = 'Green')
        ax.set_xlabel('lag')
        ax.set_ylabel('lookback')
        ax.set_zlabel(metric)
        counter+=1
    plt.show()
        
# Model assessment
def model_assess(params): 
    """
    Assess a single model
    
    Arguments:
        params: hyperparameter set
	Returns:
        'model': model
        'history': fitted model results
        'confusion_matrix': confusion matrix
        'report': classification report
        'predictions': the deterministic features for the prediction timestep
        'train_X': training features
        'train_y': training targets
        'test_X': testing features
        'test_y': testing targets
        'y_label': testing labeled values
        
	"""
    start_time = time.time()
    # assign number of features, targets and class weights
    X, y, dft = to_supervised(datasub.iloc[:,[0,2,3,1,5]],rano_clim.loc[:,params['covariates']],kian_clim.loc[:,params['covariates']],params['lookback'], params['lag'],'fruit')   
    # split dataset
    train_X, test_X, y_train, y_test, train_dft, test_dft = train_test_split(np.array(X), np.array(y), np.array(dft), test_size=DATA_SPLIT_PCT, random_state=params['seed'],stratify =np.array(y))

    if params['hs'] == 'hidden':
        model = build_hrnn(params)
    #    model.summary()
    elif params['hs'] == 'stacked':
        model = build_srnn(params)
       # model.summary()
    else:
        print('architecture not satisfied')
        exit()
    
    results = model.fit(train_X, y_train, 
                       epochs = int(params['epochs']), 
                       batch_size = int(params['batch']),
                       verbose = 2,
                       shuffle=False)
    
    y_prob = model.predict(test_X)
    y_pred = np.random.binomial(1, y_prob)
    loss = log_loss(y_test, y_prob)
    cm = confusion_matrix(y_test,y_pred) # generate confusion matrix
    class_rep = class_report(y_test,y_pred) # generate classification reports
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    fig, axis = plt.subplots(1,4,figsize=(18,4))
    plt.subplot(1,4,1)
    confusion_mat(y_test,y_pred, LABELS = LABELS, normalize = 'true')
    plt.subplot(1,4,2)
    confusion_mat(y_test,y_pred, LABELS = LABELS, normalize = None)
    plt.subplot(1, 4, 3) 
    roc_plot(y_test, y_prob) # roc curve
    plt.subplot(1, 4, 4)  
    pr_plot(y_test, y_prob) # precision recall curve
    plt.show()
    fig.tight_layout() 
    
    class_rep = class_report(y_test,y_pred) # generate classification report
    
    
    # add y and ypred to the curent covariate features
    test_dft = np.column_stack((test_dft, y_test, y_pred))
    
    print('took', (time.time()-start_time)/60, 'minutes') # print the time lapsed 
    #return the relevant information for comparing hyperparameter trials
    return {'model': model,
            'history': results,
            'confusion_matrix': cm, 
            'report': class_rep, 
            'predictions': test_dft,
            'train_X': train_X,
            'train_y': y_train,
            'test_X': test_X,
            'test_y': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'evals': [loss, pr_auc, roc_auc],
            'params': params
            }
 
def model_assess_cw(params): 
    """
    Assess a single model
    
    Arguments:
        params: hyperparameter set
	Returns:
        'model': model
        'history': fitted model results
        'confusion_matrix': confusion matrix
        'report': classification report
        'predictions': the deterministic features for the prediction timestep
        'train_X': training features
        'train_y': training targets
        'test_X': testing features
        'test_y': testing targets
        'y_label': testing labeled values
        
	"""
    start_time = time.time()
    # assign number of features, targets and class weights
    train_X, train_y, train_dft = to_supervised(train_fr.iloc[:,[0,2,3,1,5]],rano_clim.iloc[:,[0,4,5,7,8]],kian_clim.iloc[:,[0,4,5,7,8]],int(params['lookback']), 'fruit',True) 
    test_X, test_y, test_dft = to_supervised(test_fr.iloc[:,[0,2,3,1,5]],rano_clim.iloc[:,[0,4,5,7,8]],kian_clim.iloc[:,[0,4,5,7,8]],int(params['lookback']), 'fruit',True)
    features = train_X.shape[2]
    targets = 1 # assign targets
 #   weights = dict(zip([0,1], [params['weights_0'], params['weights_1']])) # get class weights
 #   class_weights = weights # assign class weights as weights
    model = hyp_rnn(params, features, targets) # generate model
    model.summary() # output model summary
    # fit the model 
    result = model.fit(train_X, train_y, 
                       epochs = int(params['epochs']), 
                       batch_size = int(params['batch_size']),
                       verbose = 2,
                       shuffle=False,
                       validation_data = (test_X, test_y))
                     #  class_weight = class_weights)
    loss, accuracy, pr_auc, roc_auc = model.evaluate(test_X, test_y)
    # make a predictions
    y_prob = model.predict(test_X)
    y_pred = np.random.binomial(1, y_prob)
    monitoring_plots(result) # plot validation plots
    cm = confusion_mat(test_y, y_pred) # plot confusion matrix
    class_rep = class_report(test_y,y_pred) # generate classification reports
    pr_auc = average_precision_score(test_y, y_prob)
    roc_auc = roc_auc_score(test_y, y_prob)
    
    # add y and ypred to the curent covariate features
    if params['n_output'] == 1:
        test_dft = np.column_stack((test_dft, test_y, y_pred))
    else:
        test_dft = np.append(test_dft, test_y, axis = 2)
        test_dft = np.append(test_dft, y_pred, axis = 2)
    print('took', (time.time()-start_time)/60, 'minutes') # print the time lapsed 
    #return the relevant information for comparing hyperparameter trials
    return {'model': model,
            'history': result,
            'confusion_matrix': cm, 
            'report': class_rep, 
            'predictions': test_dft,
            'train_X': train_X,
            'train_y': train_y,
            'test_X': test_X,
            'test_y': test_y,
            'y_pred': y_pred,
            'evals': [pr_auc, roc_auc]
            }

def model_assess_cv(params): 
    """
    Assess a single model
    
    Arguments:
        params: hyperparameter set
	Returns:
        'model': model
        'history': fitted model results
        'confusion_matrix': confusion matrix
        'report': classification report
        'predictions': the deterministic features for the prediction timestep
        'train_X': training features
        'train_y': training targets
        'test_X': testing features
        'test_y': testing targets
        'y_label': testing labeled values
        
	"""
    start_time = time.time()
    # assign number of features, targets and class weights
 #   X, y, dft = to_supervised(train_val_fr.iloc[:,[0,2,3,1,5]],rano_clim.iloc[:,[0,4,5,7,8]],kian_clim.iloc[:,[0,4,5,7,8]],params['lookback'], 'fruit',True)   
    train_X, train_y, train_dft = to_supervised(train_fr.iloc[:,[0,2,3,1,5]],rano_clim.iloc[:,[0,4,5,7,8]],kian_clim.iloc[:,[0,4,5,7,8]],params['lookback'], 'fruit',True)   
    test_X, test_y, test_dft = to_supervised(test_fr.iloc[:,[0,2,3,1,5]],rano_clim.iloc[:,[0,4,5,7,8]],kian_clim.iloc[:,[0,4,5,7,8]],int(params['lookback']), 'fruit',True)
    
    features = train_X.shape[2]
    targets= params['n_output'] # set targets
    weights = dict(zip([0,1], [params['weights_0'], params['weights_1']]))
    model = KerasClassifier(build_fn=hyp_rnn_cv, 
                            features = features, targets = targets, 
                            lookback = params['lookback'], mtype = params['mtype'], 
                            layers = params['layers'], neurons_n = params['neurons_n'],
                            hidden_n = params['hidden_n'], d_rate = params['d_rate'], 
                            lr_rate = params['lr_rate'], verbose=2, class_weight = weights, 
                            epochs = params['epochs'], batch_size = params['batch_size']) # build model
    results = model.fit(train_X, train_y)
    # # make a predictions
    y_prob = model.predict_proba(test_X)[:,1]
    y_pred = np.random.binomial(1, y_prob)
 #   monitoring_plots(result) # plot validation plots
    cm = confusion_mat(test_y, y_pred) # plot confusion matrix
    class_rep = class_report(test_y,y_pred) # generate classification reports
    pr_auc = average_precision_score(test_y, y_prob)
    roc_auc = roc_auc_score(test_y, y_prob)
    
    # add y and ypred to the curent covariate features
    if params['n_output'] == 1:
        test_dft = np.column_stack((test_dft, test_y, y_pred))
    else:
        test_dft = np.append(test_dft, test_y, axis = 2)
        test_dft = np.append(test_dft, y_pred, axis = 2)
    print('took', (time.time()-start_time)/60, 'minutes') # print the time lapsed 
    #return the relevant information for comparing hyperparameter trials
    return {'model': model,
            'history': results,
            'confusion_matrix': cm, 
            'report': class_rep, 
            'predictions': test_dft,
            'train_X': train_X,
            'train_y': train_y,
            'test_X': test_X,
            'test_y': test_y,
            'y_pred': y_pred,
            'evals': [pr_auc, roc_auc]
            }
# Permutation analysis
def pi_info(cm, report, evals, feature, lookback):
    """
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
    datarow = [feature,lookback,report.iloc[2,0]] + evals # add lookback and feature and accuracy, loss, pr_auc, roc_auc (5)
    datarow.extend(report.iloc[3,0:3].tolist()) # overall precision, recall and f1 score (3)
    datarow.extend(report.iloc[0:2,2].tolist()) # add categorical f1 score (2)
    datarow.extend(report.iloc[0:2,0].tolist()) # add categorical precision (2)
    datarow.extend(report.iloc[0:2,1].tolist()) # add categorical recall (2)
    conmat = np.reshape(cm,(1,4)) # add cofusion matrix values (16)
    datarow.extend(conmat.ravel().tolist())
    return(datarow)

def perm_feat(start_index, end_index, t, eval_X):
    """
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
    
def perm_assess(model, X_reshape, y, cv = False): 
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
    y_pred = np.random.binomial(1, y_prob)
    loss = log_loss(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    #y_pred= y_prob.argmax(axis=-1)
    # get confusion matrix
    cm = confusion_matrix(y,y_pred)
    # get classification report
    class_rep = classification_report(y,y_pred, zero_division = 0, output_dict = True)
    class_rep = DataFrame(class_rep).transpose()
    return {'confusion_matrix': cm, 
            'report': class_rep,
            'evals': [loss, pr_auc, roc_auc]}
def perm_importance(model, og_cm, og_report, og_evals, eval_X, y, lag = 0, feature_names = ['soilmoist.mean'], filename = 'perm_imp_default', cv = False):
    # list column names 
    start_time = time.time()
    colnames = ['feature','lookback','accuracy','loss','pr_auc','roc_auc','precision','recall','f1_score','f1_0','f1_1',
            'precision_0','precision_1','recall_0','recall_1', 'TN','FP','FN','TP']
    df = pd.DataFrame(columns = colnames)
    datarow = pi_info(og_cm, og_report, og_evals, 'original', np.nan)
    df.loc[len(df.index)]= datarow
    
    # for each lookback period
    for t in range(0,(eval_X.shape[1])):
        counter = 0 # start counter for feature name
        for f in range(eval_X.shape[2]):
            # do single permutations
            eval_X_copy = perm_feat(f,f+1,t,eval_X)
            perm_output = perm_assess(model, eval_X_copy, y, cv)
            datarow = pi_info(perm_output['confusion_matrix'], perm_output['report'], perm_output['evals'],feature_names[counter], (eval_X.shape[1]-t-lag))
            df.loc[len(df.index)]= datarow
            print(feature_names[counter], t-lag)
            counter +=1
    df['loss_diff'] = df['loss'] - df['loss'][0]
    df['acc_diff'] = df['accuracy'] - df['accuracy'][0]
    df['roc_diff'] = df['roc_auc'] - df['roc_auc'][0]    
    df['pr_diff'] = df['pr_auc'] - df['pr_auc'][0]
    df.to_csv(filename + '.csv')
    print('took', (time.time()-start_time)/60, 'minutes') # print the time lapsed 
    return df

def pi_plot(df, metrics):
    '''

    Parameters
    ----------
    df : permutation importance dataframe
    metric : metric to plot to measure permutation importance

    Returns
    -------
    None.

    '''
    features = df['feature'].unique()
    features = np.delete(features,0)
    nf = len(features) # number of features
    nm = len(metrics)
    counter =1
    fig, axs = plt.subplots(nf, nm, figsize = (4*nm, 4*nf))
    for i in range(nf):
        for metric in metrics:
            plt.subplot(nf,nm, counter)
            sub_df = df[df['feature'] == features[i]]
            plt.bar(x = sub_df['lookback'], height = sub_df[metric])
            plt.title(features[i])
            plt.xlabel('days prior')
            plt.ylabel(metric)
            counter+=1
    fig.tight_layout()  
    plt.show()
        
def pimp_model(rnn_perm, assessed, trials, seed, bid, atype):
    """
    Run and assess model using only the most important feature
    
    Parameters
    ----------
    rnn_perm : permutation importance results
    assessed : assessmenet output for best model
    trials : hyperopt trials
    seed : seed for hyperopt trials
    bid : best model index
    atype : architecture type ('VRNN' or 'ENDE')

    Returns
    -------
    None.

    """
    best_look = (int(rnn_perm.iloc[0,1]) + 1)*-1
    X_train = assessed['train_X']
    X_train = np.copy(X_train[:,best_look,0:4])
    X_train = X_train[:,newaxis,:]
    X_test = assessed['test_X']
    X_test = np.copy(X_test[:,best_look,0:4])
    X_test = X_test[:,newaxis,:]
    y_train = assessed['train_y']
    y_test = assessed['test_y']
    
    best_trial = trials.results[bid]['params']
    best_trial['lookback'] = 1
    weights = dict(zip([0,1,2,3], [best_trial['weights_0'], best_trial['weights_1'], best_trial['weights_2'], best_trial['weights_3']])) # optimize class weights
    if atype == 'VRNN':
        class_weights = weights # assign class weights as weights
        sample_weights = None
        early_stopping = EarlyStopping(patience= best_trial['epochs'], monitor='val_f1_score', mode = 'max', restore_best_weights=True, verbose=0)
        model = hyp_rnn_nest(params =best_trial, features =4, targets=4)
    else:
        class_weights = None 
        total = sum(weights.values()) # get the sum of the weights to normalize
        sample_weights = {ky: val / total for ky, val in weights.items()} # get the sample weight values
        sample_weights = get_sample_weights(y_train, weights) # generate the formatted sample weights 
        early_stopping = F1EarlyStopping(validation_data=[X_test, y_test], train_data=[X_train, y_train], patience= best_trial['epochs'])
        model = hyp_ende_nest(params =best_trial, features =4, targets=4)  
    model.summary() # output model summary
    
# fit the model
    result = model.fit(X_train, y_train, 
                           epochs = best_trial['epochs'], 
                            batch_size = best_trial['batch_size'],
                            verbose = 2,
                            shuffle=False,
                            validation_data = (X_test, y_test),
                            sample_weight = sample_weights,
                            class_weight = class_weights,
                            callbacks = [early_stopping])
    
    # make a predictions
    y_prob = model.predict(X_test)
    y_pred = to_label(y_prob)
    y_label = to_label(y_test)
    
    if atype == 'VRNN':
        monitoring_plots(result) # plot validation plots
    else:
        monitoring_plots(result, early_stopping)
    confusion_mat(y_label, y_pred) # plot confusion matrix
    class_report(y_label,y_pred) # generate classification reports    
    # add y and ypred to the curent covariate features
    t_features = DataFrame(assessed['predictions'], columns = datasub.columns.values[(7+4):(7+18)].tolist() +['y','y_pred'])
    t_features['y_pred'] = y_pred
    t_prop = daily_dist(t_features)
    daily_dist_plot(t_prop)
    
def model_postnalysis(seed, atype, mode  = None):
    '''
    Function to assess the best model results, run permutation analysis and run model with only important values w/ corresponding results

    Parameters
    ----------
    seed : Hyperparameter seed
    atype : architecture type ('VRNN' or 'ENDE')
    
    Returns
    -------
    Loss graph, f1 graph, classification report, confusion matrix and daily behavioral distributions for best model for the seed and importance variable model
    top 10 most important features according to permutation importance for the best model
    
    Best model f1 graph

    '''
    if atype == 'VRNN':
        if mode == 'bonly':
            trials = joblib.load('vrnn_bonly_vv_trials_seed'+str(seed)+'.pkl')
        else:
            trials = joblib.load('vanilla_rnn_vv_trials_seed'+str(seed)+'.pkl')
        rnn_df = read_csv('vrnn'+str(seed)+'.csv', header = 0, index_col = 0)
        
    else:
        trials = joblib.load('ende_vv_trials_seed'+str(seed)+'.pkl')
        rnn_df = read_csv('ende'+str(seed)+'.csv', header = 0, index_col = 0)
    
    bid = rnn_df.loc[rnn_df.val_f1 == max(rnn_df['val_f1'])].index.values[0]
   
    assessed = model_assess(trials.results[bid]['params'], atype)
    rnn_prop = daily_dist(assessed['predictions'])
    daily_dist_plot(rnn_prop)
    
    if mode != 'bonly':
        rnn_perm = perm_importance(assessed['model'],assessed['confusion_matrix'], assessed['report'], assessed['test_X'], assessed['y_label'])    
        rnn_perm['f1_diff'] = abs(rnn_perm['f1_score']-rnn_perm['f1_score'][0])
        rnn_perm.sort_values(by = ['f1_diff'], axis=0, ascending=False, inplace = True)
        print(rnn_perm.iloc[0:10,[0,1,34]])
        rnn_perm.to_csv(str(atype)+str(seed)+'_perm_df.csv')
        
        # return {'rnn_perm': rnn_perm,
        #         'trials': trials,
        #         'assessed': assessed,
        #         'bid': bid
        #         }
        pimp_model(rnn_perm,assessed, trials, seed, bid, atype)
 
def model_pipeline_valid(params):
   # start_time = time.time()
    # create dataset 
    X, y, dft = to_supervised(datasub.iloc[:,[0,2,3,1,5]],rano_clim.loc[:,params['covariates']],kian_clim.loc[:,params['covariates']],params['lookback'], params['lag'],'fruit')   
    # split dataset
    X_train_scaled, X_test_scaled, y_train, y_test, dft_train, dft_test = train_test_split(np.array(X), np.array(y), np.array(dft), test_size=DATA_SPLIT_PCT, random_state=params['seed'],stratify =np.array(y))
    X_train_scaled, X_valid_scaled, y_train, y_valid, dft_train, dft_valid = train_test_split(X_train_scaled, y_train, dft_train, test_size=DATA_SPLIT_PCT, random_state=SEED, stratify = y_train)

    if params['hs'] == 'hidden':
        model = build_hrnn(params)
    #    model.summary()
    elif params['hs'] == 'stacked':
        model = build_srnn(params)
       # model.summary()
    else:
        print('architecture not satisfied')
        exit()
    
        print('architecture not satisfied')
        exit()
        
    history = model.fit(X_train_scaled, y_train, 
                         epochs = int(params['epochs']), 
                         batch_size = int(params['batch']),
                         verbose = 2,
                         shuffle=False,
                         validation_data = (X_valid_scaled,y_valid))
     
    plot_monitor(history) # monitoring plots

# trials = joblib.load('phen_basic_gru_sho_watertemp_405.pkl')


# trial_df = trial_correg_plots(trials, 'SHO GRU soil moist temp 405')

# triplot3d(trial_df[0])
# kde_comp_mm(trial_df[0], np.r_[14,16],['val_acc','val_roc','val_pr'],'SHO GRU soil moist temp 405')

# # get index of trial with highest validation score
# bid = trial_df[0].nlargest(1,'val_pr').index.values[0]
# params = trials.results[bid]['params']
# params['epochs'] = 500
# model_pipeline_valid(params)
# params['epochs'] = 1000
# model_pipeline_valid(params)
# assessed = model_assess(params)

# p_df = perm_importance(assessed['model'],assessed['confusion_matrix'], assessed['report'],assessed['evals'],
#                          assessed['test_X'],assessed['test_y'], assessed['params']['lag'], 
#                          list(assessed['params']['covariates'][1:]),'pi_gru_sho_moisttemp_405_loss_' + str(bid))

# pi_plot(p_df,['loss_diff','acc_diff','roc_diff','pr_diff'])