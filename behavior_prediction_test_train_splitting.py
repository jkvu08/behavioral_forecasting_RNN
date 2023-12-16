# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:40:03 2021
Split the training and testing data using single holdout method
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
os.chdir("C:\\Users\\Owner\\Documents\\Dissertation\\Data")
# import the datafile
dataset = pd.read_csv('behave_formatted_2021_10_16.csv', header=0, index_col = 0)
dataset_ss = pd.read_csv('ss_lstm_formatted_2021_07_13.csv',header = 0, index_col = 0)
dataset.head()  # look at first 6 rows

# separate out the testing dataset
##########################
#### SEPARATE BY FOCAL#### 
##########################
test_focals = ['FIG','YAM','BLU','GRY','LTB','PUR','TGK','PKL','ZPH','SPD','RAP','MKL']

test_focals = random.sample(set(dataset.focal),6) # take a random sample of six individuals (these individuals will make up the test set)
# split testing from training based on the focals 
test = dataset[dataset['focal'].isin(test_focals)] # subset the dataset by the individuals selected for the test set
train = dataset[~dataset['focal'].isin(test_focals)] # subset the dataset by the individuals that were not selected for the test set

test.to_csv('kian_testset_focal_2022.csv') # save the test and train sets
train.to_csv('kian_trainset_focal_2022.csv') # save the test and train sets

# repeat for sangasanga
test_focals1 = random.sample(set(dataset_ss.focal), 4)
test_ss = dataset_ss[dataset_ss['focal'].isin(test_focals1)]
train_ss = dataset_ss[~dataset_ss['focal'].isin(test_focals1)]

test_ss.to_csv('ss_testset_focal.csv')
train_ss.to_csv('ss_trainset_focal.csv')

##########################
#### SEPARATE BY YEAR #### 
##########################
test1 = dataset[dataset['year'] >= 2018] # subset out the years 2018 and 2019 for the test set
train1 = dataset[dataset['year'] < 2018] # subset out the years before 2018 for the train set 
test1.to_csv('vv_testset_year.csv') # save the results
train1.to_csv('vv_trainset_year.csv') # save the results

# repeat for sangasanga 
test2 = dataset_ss[dataset_ss['year'] >= 2018]
train2 = dataset_ss[dataset_ss['year'] < 2018]
test2.to_csv('ss_testset_year.csv')
train2.to_csv('ss_trainset_year.csv')

# separate the independent and dependent variables
test_X = test1.iloc[:,11:] 
test_Y = test1.iloc[:,7:11]

train_X = train2.iloc[:,11:]
train_Y = train2.iloc[:,7:11]

# examine the stratification of the datasets check if the distributions are the same
train_Y.value_counts(normalize = True)*100 # check if the data is balanced
test_Y.value_counts(normalize = True)*100 # check balanced


sample_size = np.array([[1,len(train_ss),len(test_ss), len(train_ss)+len(test_ss)]])
for i in range(2,25):
    test_sub = test_ss[test_ss['track_position'] >= i]
    train_sub= train_ss[train_ss['track_position'] >= i]
    sub_line = np.array([[i,len(train_sub),len(test_sub), len(train_sub)+len(test_sub)]])
    sample_size = np.concatenate((sample_size,sub_line), axis = 0)
    
np.savetxt("ss_sample_size.csv", sample_size, delimiter = ",")

##########################
#### SEPARATE BY ENTRY#### 
##########################

# separate features and lables
X = dataset.iloc[:,11:]
Y = dataset.iloc[:,7:11]

X_train, X_test, y_train, y_test = train_test_split(dataset, Y, test_size=0.20, random_state=1, stratify = Y)
y_train.value_counts(normalize = True)*100
y_test.value_counts(normalize = True)*100

X_train.to_csv('vv_trainset_entry.csv')
X_test.to_csv('vv_testset_entry.csv')
