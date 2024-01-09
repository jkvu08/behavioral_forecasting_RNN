# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet
"""
# Load libraries
import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import joblib
import seaborn as sns
import glob

# Load and Prepare Dataset
# set working directory
os.chdir("C:\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\behavioral_forecasting_RNN\\outputs\\"

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
    avg_results = results.avg_eval.to_list() # convert all all avg results for each hyperopt experiment into list                
    avg_results = pd.concat(avg_results, axis = 1).transpose() # convert each avg result into dataframe and transpose so each row represents a hyperoptimization experiment
    df = pd.concat([params_df, avg_results], axis = 1) # concatenate experiment parameters and metrics into a single dataframe
    # set nan value for hidden neurons that hyperopt drew but did not use in the model based on the hidden layers implemented 
    df.loc[df['hidden_layers'] < 1, 'hidden_n0'] = np.NaN 
    df.loc[df['hidden_layers'] < 2, 'hidden_n1'] = np.NaN 
    df.sort_index(inplace = True) # ensure the indices are ordered
    df['epochs'] = df['epochs'].astype('int64')
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
        if metric in ['hidden_n0','hidden_n1']:
            plt.plot(df.index, df[metric], marker='.', markersize = 3, label = metric) # plot the metric by the index
        else:
            plt.plot(df.index, df[metric], label = metric) # plot the metric by the index
        plt.ylabel(metric) # label the metric 
        counter +=1 # increase the counter
    fig.supxlabel('runs') # add x axis label
    fig.tight_layout() 
    return fig

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
    path : str, 
        directory location to save output file 
    filename: str, 
        filename prefix for output file
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

def get_ci_df(df,threshold):
    """
    extract credible interval limits for each metric in a dataframe

    Parameters
    ----------
    df : dataframe, metric values 
    threshold : float, 0-1
        the credible interval threshold (alpha)

    Returns
    -------
    lp : series, 
        lower credible intervals
    up: series, 
        upper credible intervals
    
    """
    # calculate alpha for credible interval
    lb = (1-threshold)/2 # alpha
    ub = 1-((1-threshold)/2) # 1- alpha
    
    # create empty list to population metrics 
    lp = []
    up = []
    
    collist = list(df.columns) # get metric columns as list
    for i in collist: # for each column
        svalues = df.loc[:,i].dropna() # extract metric column and drop na values
        n = svalues.count() # get number of non na values
        # pull the lower and upper limit indices that correspond to the credible interval cut off
        llim = int(np.floor(n*lb))-1
        ulim = int(np.ceil(n*ub))-1
        svalues = svalues.sort_values(ignore_index=True) # sort values
        # pull credible interval values
        lp.append(svalues[llim])
        up.append(svalues[ulim])
    # output as series
    lp = pd.Series(lp, index = collist)
    up = pd.Series(up, index = collist)
    return lp, up

def get_ci(ary, threshold):
    """
    extract credible interval limits for each metric in an array

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
    for i in range(ary.shape[1]): # for each metric
        svalues = np.sort(ary[:,i]) # sort the values of metrics
        svalues = svalues[~np.isnan(svalues)] # get rid of na values
        n = len(svalues)
        # pull the lower and upper limit indices that correspond to the credible interval cut off
        llim = int(np.floor(n*lci))-1
        ulim = int(np.ceil(n*uci))-1
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
        directory where hyperopt files are saved
    modelname : str, 
        model filename prefix
    params : list, 
        parameters/hyperparameters to evaluate
    ci: numeric, 
        credible interval threshold. Default is 0.90.
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
    colnames = loss + metrics
    dflist = [] # empty list for dataframes to combine of hyperopt outputs
    for file in glob.glob(path + modelname + '*.pkl'): # load all the hyperopt trials using the list of filenames
        trials = joblib.load(file) # load hyperopt trial file
        trial_df = trials_to_df(trials) # convert hyperopt experiment results to a dataframe
        trial_df.to_csv(file[:-4]+'_results.csv')
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
    sumval = pd.concat([minval, medval]) # concatenate the median and min values
    format_values = [] # create empty list to populate format values
    
    # generate formated values 'med or min (lower CI, upper CI)'
    for i in range(len(colnames)):
        if trial_df[colnames[i]].dtype == 'int64': # format as integer 
            text  = str(sumval[i].astype('int64')) + ' (' +str(lci[i].astype('int64')) +','+ str(uci[i].astype('int64')) + ')'
        else: # keep as float
            text  = str(sumval[i]) + ' (' +str(lci[i]) +','+ str(uci[i]) + ')'
        format_values.append(text)
    # convert to dataframe
    output = pd.DataFrame(format_values, 
                          index = colnames, 
                          columns= [modelname]) 
    return output # output values and intervals
   
def sum_function(df,filename, path = None):
    """
    Calculates summary statistics and credible intervals of the performance metrics
    across all trials for a particular model architecture

    Parameters
    ----------
    df : dataframe,
        performance metrics
    filename : str, optional 
        filename  
    path : str, optional
        directory to save file. Default is None
        

    Returns
    -------
    summary_df : dataframe,
        summary statistics of model performance with credible intervals.

    """
    # calcuate summary statistics
    grand_mean = df.mean(axis = 0) # calcualte grand median
    grand_median = df.median(axis = 0) # calcualte grand median
    grand_sd = df.std(axis = 0) # calculate grand standard deviateion
    grand_mad = df.mad(axis = 0) # calculate grand standard deviateion
    
    # calcuate credible intervals 
    l50, u50 = get_ci(df.to_numpy(), 0.50) # get 50% credible intervals 
    l80, u80 = get_ci(df.to_numpy(), 0.80) # get 80% credible intervals
    l90, u90 = get_ci(df.to_numpy(), 0.90) # get 90% credible intervals
    l95, u95 = get_ci(df.to_numpy(), 0.95) # get 95% credible intervals
    
    # concatenate credible intervals
    ci_df = pd.DataFrame([l95, u95, l90, 
                          u90, l80, u80, 
                          l50, u50], 
                         columns = grand_mean.index)
    
    # concatenate summary statistics with credible intervals
    summary_df = pd.concat([grand_mean, grand_sd, 
                            grand_median, grand_mad,
                            ci_df.transpose()],
                            axis = 1) # generate summary table 
    
    # name the columns
    summary_df.columns = ['mean','sd','median', 
                          'mad','lci95','uci95',
                          'lci90','uci90','lci80',
                          'uci80','lci50','uci50']
    
    summary_df['model'] = filename # add model identifier
    summary_df = summary_df.iloc[:, np.r_[-1, 0:(summary_df.shape[1]-1)]]  # reorder dataframe so model identifier is the first column
    if path != None:
        summary_df.to_csv(path+filename+'_performance_summary.csv') # save output
    return summary_df

def rhat_calc(values):
    """
    Calculate Gelman-Rubin rhat estimate of parameter convergence across multiple experiments

    Parameters
    ----------
    values : list,
       Set of the parameters selected in each hyperoptimization experiment. Assumes that the same number of trials were ran within each experiment   
    Raises
    ------
    Exception
        Too few models to compare

    Returns
    -------
    rhat : Gelman-Rubin rhat value

    """
    n_exp = len(values) # number of experiments
    if n_exp > 1:
        var_list = [] # list of variances lists
        mean_list = [] # list of means
        # generate mean and variance for each experiment
        for i in range(n_exp):
            exp_var = values[i].var(axis = 0) # get trial variance 
            exp_mean = values[i].mean(axis = 0) # get trial variance 
            var_list.append(exp_var)  # add to variance list
            mean_list.append(exp_mean)  # add to variance list     
        # convert lists to array
        var_list = np.array(var_list)
        within_mean = np.array(mean_list)
        within_var = np.mean(var_list) # calculate within variance
        full_values = pd.concat(values, ignore_index = True) # concatenate all values into single file
        full_values.dropna(inplace =True) # drop all na values since some parameters were not used in all trials (i.e., hidden neurons)
        n_trials = int(len(full_values)/n_exp) # calculate number of non-na parameter draws per experiment, assuming same number of non-na parameters were drawn in each experiment for the calculation 
        grand_mean = full_values.mean(axis = 0) # calculate grand mean
        # calculate the between exp variance
        bvalues = (within_mean - grand_mean)**2 # get difference between trial means and grand mean and square
        bvalues = sum(bvalues) # sum together
        between_var = n_trials/(n_exp-1)*bvalues # square and multiply by data factor
        rhat = (((n_trials-1)/n_trials)*within_var + (1/n_trials)*between_var)/within_var # calculate rhat   
    else:
        raise Exception ('too few experiments to compare')  

    return rhat
        
# calculate rhat to check for convergence
def convergence_sum(modelname, path, params, burnin = 200, maxval = 1000):
    """
    Calculates Gelman-Rubin rhat summary statistics and credible intervals of the performance metrics
    across all trials for a particular model architecture.

    Parameters
    ----------
    modelname : str, 
        model filename prefix
    path : str,
        directory where hyperopt files are saved
    params : list, 
        parameters/hyperparameters to evaluate
    burnin : int, 
        number of initial experiments to discard. Dafault is 200.
    maxval : int, 
        maximum number of experiments to compare. Default is 1000.

    Returns
    -------
    exp : list,
        hyperopt experiment output dataframes subset by burnin and maxeval
    df : dataframe,
        single dataframe of outputs for all hyperopt experiments of a single model type
        
    summary_df : dataframe,
        summary statistics of model performance with credible intervals and Gelman-Rubin rhat.

    """
    # load in all files of the model type
    exp = [] # create empty list to populate with files
    for file in glob.glob(path + modelname +'*.pkl'): # for files with this prefix
        file = joblib.load(file) # load each file 
        trial_df = trials_to_df(file)  # convert to dataframe
        trial_df = trial_df[params] # subset params of interest only
        trial_df = trial_df.iloc[burnin:maxval,] # drop burnin draws and subsete to maxval to ensure comparing same number of trials across experiments 
        exp.append(trial_df) # add trial dataframe to list
    # calculate rhat 
    rhat_list = [] # empty list for rhat
    for i in params: # for each parameter
        params_values = [] # create list for subseting parameter column from trial dataframes
        for j in range(len(exp)): # for each trial
            params_values.append(exp[j][i]) # append the paramater column
        rhat = rhat_calc(params_values) # calcualate the rhat
        rhat_list.append(rhat) # append rhat to list of rhats
    
    df = pd.concat(exp, axis = 0, ignore_index = True) # concatenate all experiments into a single dataframe
    summary_df = sum_function(df, modelname)
    summary_df['rhat'] = rhat_list
    summary_df.to_csv(path + modelname + '_'+str(burnin)+'_summary.csv') #save output
    return exp, df, summary_df # return summary table 

def chain_plots(trials, metric, ax = None):
    """
    Overlay of the progression of a particular matric throughour the hyperopt experiments
    
    Parameters
    ----------
    trials : list,
        hyperopt experiment output dataframes subset by burnin and maxeval
    metric : str,
        parameter/metric to monitor
    ax : list, optional
        subplot location. The default is None.

    Returns
    -------
    None

    """
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']
    for i in range(len(trials)):    
        sns.lineplot(ax = ax, data = trials[i], x = trials[i].index, y = metric, alpha = 0.5, color = colors[i])

def hist_plots(trials, metric, ax = None):
    """
    Overlay of the histograms of a particular metric from the hyperopt experiments
    
    Parameters
    ----------
    trials : list,
        hyperopt experiment output dataframes subset by burnin and maxeval
    metric : str,
        parameter/metric to monitor
    ax : list, optional
        subplot location. The default is None.

    Returns
    -------
    None

    """
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']
    for i in range(len(trials)):   
        sns.histplot(ax = ax, data = trials[i],x = metric, bins = 20, alpha =0.5, color = colors[i])

def kde_plots(trials, metric, ax= None):
    """
    Overlay of kernel density distribution of a particular metric against validation loss from the hyperopt experiments
    
    Parameters
    ----------
    trials : list,
        hyperopt experiment output dataframes subset by burnin and maxeval
    metric : str,
        parameter/metric to monitor
    ax : list, optional
        subplot location. The default is None.

    Returns
    -------
    None

    """
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']
    for i in range(len(trials)):  
        sns.kdeplot(ax = ax, data = trials[i], x = metric, y = 'val_loss', shade = True, legend = False, color = colors[i], alpha =0.5)
    
def loss_plots(trials):
    """
    Visualization of loss in hyperopt experiments
    
    Parameters
    ----------
    trials : list,
        hyperopt experiment output dataframes subset by burnin and maxeval
    
    Returns
    -------
    fig : overlayed progression and histogram of the losses

    """
    #n = int(combo_df.shape[0]/3)
    fig, ax = plt.subplots(2,2,gridspec_kw={'width_ratios': [2, 1]}, figsize = [10,3])
    chain_plots(trials, 'val_loss', ax[0,0])
    hist_plots(trials, 'val_loss', ax[0,1])
    chain_plots(trials, 'train_loss', ax[1,0])
    hist_plots(trials, 'train_loss', ax[1,1])
    fig.tight_layout()
    return fig

def parameter_plots(trials, metrics):
    """
    Visualization of parameters in hyperopt experiments
    
    Parameters
    ----------
    trials : list,
        hyperopt experiment output dataframes subset by burnin and maxeval
    metrics : list,
        list of parameters/metrics to monitor

    Returns
    -------
    fig : overlayed progression, histogram, and kernel density plots of the parameters

    """
    fig, ax = plt.subplots(len(metrics),3, gridspec_kw={'width_ratios': [2, 1, 1]},figsize = (10,12))
    for i in range(len(metrics)):
        chain_plots(trials, metrics[i], ax[i,0])
        hist_plots(trials, metrics[i], ax[i,1])
        kde_plots(trials, metrics[i], ax[i,2])
    fig.tight_layout()
    return fig
    
# start annotating here 
def trial_chains_output(modelname, path, params, monitor = ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc'], burnin = 200, maxval = 1000):
    """
    Wrapper to generate summary statistics, visualize outputs from hyperopt experiments and output to a pdf.
    
    Parameters
    ----------
    modelname : str, 
        model filename prefix
    path : str,
        directory where hyperopt files are saved
    params : list, 
        parameters/hyperparameters to evaluate
    monitor : list, optional
        Loss and performance metrics. The default is ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc'].
    burnin : int, 
        number of initial experiments to discard. Dafault is 200.
    maxval : int, 
        maximum number of experiments to compare. Default is 1000.

    Returns
    -------
    None

    """
    metrics = monitor + params 
    # identify the float metrics for rounding later 
    float_met = ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc','dropout_rate','weights_0']
    # get experiment outputs as list and dataframe, and summary table
    trials, combined_df, summary_df = convergence_sum(modelname = modelname, 
                                                      path = path,
                                                      params = metrics, 
                                                      burnin = burnin, 
                                                      maxval = maxval)
    # round float metrics
    summary_df.loc[summary_df.index.isin(float_met),:] = round(summary_df.loc[summary_df.index.isin(float_met),:],2) 
    summary_df.loc[:,['mean','sd','median','mad','rhat']] = round(summary_df.loc[:,['mean','sd','median','mad','rhat']],2)
    # format summary table 
    sum_tab = summary_df[['mean','sd','median','mad','lci95','uci95','rhat']]
    # format credible interval table
    ci_tab = summary_df[['median','mad','lci95','uci95','lci90','uci90','lci80','uci80','lci50','uci50']]
    
    with PdfPages(path + modelname +'_multichain_results.pdf') as pdf:
        # generate loss plots
        lossplots = loss_plots(trials)
        pdf.savefig(lossplots) # save figure
        plt.close() # close page
        
        # generate parameter progression plots
        paramplots = parameter_plots(trials,params)
        pdf.savefig(paramplots) # save figure
        plt.close() # close page
        
        # generate subplots to organize table data 
        fig, ax = plt.subplots(3,1, sharex = False, sharey = False, figsize=(10,12))
        # add train v validation loss plot
        sns.regplot(ax = ax[0], 
                    data = combined_df, 
                    x = 'train_loss', 
                    y = 'val_loss', 
                    fit_reg = True,
                    color = 'orange')
        # add summary table 
        ax[1].table(cellText = sum_tab.values,
                    colLabels = sum_tab.columns, 
                    rowLabels = sum_tab.index,
                    loc = 'center')
        ax[1].axis('tight') 
        ax[1].axis('off')
        
        # add the credible interval table
        ax[2].table(cellText = ci_tab.values,
                    colLabels = ci_tab.columns, 
                    rowLabels = ci_tab.index,
                    loc = 'center')
        ax[2].axis('tight') 
        ax[2].axis('off')
        pdf.savefig() # save figure
        plt.close() # close page
