# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:40:03 2021
Split the training and testing data using single holdout method by focal individual and year
@author: Jannet

Input: Processed data
Output: Training/testing data
"""
### Load in data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random

######################
#### Data Import #####
######################
# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

# import the formatted behavioral data
# formatted using the behavior_preprocessing code
dataset = pd.read_csv('behavior_formatted.csv', 
                      header=0, 
                      index_col = 0)

dataset.head()  # look at first 6 rows
dataset.shape[0] # get number of records to start (133825 records)

# get rid of data from 2010 since that was the first year data was collected and protocol may be inconsistent
dataset = dataset[dataset['year'] > 2010]
dataset.shape[0] # get number of remaining records (125093 records)

############################
#### SPLIT DATA BY FOCAL#### 
############################
# get number of focal individuals
vvn = len(dataset[dataset['fragment']==0]['focal'].unique()) # in Vatovavy
ssn = len(dataset[dataset['fragment']==1]['focal'].unique()) # in Sangasanga
vvn # 24 unique individuals
ssn # 17 unique individuals

# encode focal individuals using factors to protect focal identity
dataset['fcode'] = pd.factorize(dataset.focal.values,
                   na_sentinel=None, 
                   sort=True)[0] 

# get unique factor for focal individuals per fragment as lists
vvfocals = list(dataset[dataset['fragment'] == 0]['fcode'].unique())
ssfocals = list(dataset[dataset['fragment'] == 1]['fcode'].unique())

# testing split proportion
split_prop = 0.6 # 60% used for training and 40% by default used for testing

# sample for test focal individuals
# take a random sample of Vatovavy individuals for testing set)
focals_vv = random.sample(vvfocals,
                          round(vvn*split_prop)) 
# take a random sample of Sangasanga individuals for testing set)
focals_ss = random.sample(ssfocals,
                          round(ssn*split_prop)) 
train_focals = focals_vv + focals_ss # merge focal lists into one

# split data into training-testing subsets based on the focal individual
train = dataset[dataset['fcode'].isin(train_focals)] # subset the dataset by the individuals selected for the train set
test = dataset[~dataset['fcode'].isin(train_focals)] # subset the dataset by the individuals that were not selected for the train set

# get proportion of data that makes up the training and testing data 
# want approx 60% of records used for training and 40% used for testing so may need to run split again to get appropriate ratio
# since sampling varied across individuals due to dropped collars, dispersal and mortality
train.shape[0]/dataset.shape[0]
test.shape[0]/dataset.shape[0]

# save the test and train sets
test.to_csv('kian_testset_focal.csv') 
train.to_csv('kian_trainset_focal.csv') # save the test and train sets

# examine the stratification of the datasets check if the distributions are the same
# subset dependent variables
test_Y = test.iloc[:,7:11]
train_Y = train.iloc[:,7:11]

train_Y.value_counts(normalize = True)*100 # check if the data is balanced
test_Y.value_counts(normalize = True)*100 # check balanced

##########################
#### SEPARATE BY YEAR #### 
##########################
# data records from 2011-2019 (9 years)
# get cumulative data records across years to see when to cut off years
# get number of records recorded per year
dataset_sum = dataset.groupby('year').size() 
# get cumulative records recorded per year
dataset_cs = np.cumsum(dataset_sum) 
# concatenate the record and cumulative records by year
dataset_sum  = pd.concat([dataset_sum, dataset_cs], 
                         axis = 1) 
# rename columns
dataset_sum.rename(columns = {0: 'count',1:'cumulative'}, 
                   inplace = True) 

# get proportion of data cumulative collected each year
dataset_sum['cum_prop'] = dataset_sum['cumulative']/dataset_sum['cumulative'].max() 

# year  count  cumulative  cum_prop                            
# 2011  16092       16092  0.128640
# 2012  16844       32936  0.263292
# 2013  16771       49707  0.397360
# 2014  13598       63305  0.506063
# 2015  14458       77763  0.621641
# 2016  13795       91558  0.731919
# 2017  14833      106391  0.850495
# 2018  10016      116407  0.930564
# 2019   8686      125093  1.000000

# sticking with ~60% training anda ~40% data split
test = dataset[dataset['year'] >= 2016] # years 2011-2015 should be used for training
train = dataset[dataset['year'] < 2016] # and years 2016-2019 should be used for testing 

# save the split
test.to_csv('kian_testset_year.csv') 
train.to_csv('kian_trainset_year.csv') 

# examine the stratification of the datasets check if the distributions are the same
# subset dependent variables
test_Y = test.iloc[:,7:11]
train_Y = train.iloc[:,7:11]

train_Y.value_counts(normalize = True)*100 # check if the data is balanced
test_Y.value_counts(normalize = True)*100 # check balanced