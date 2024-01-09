# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:09:47 2021

@author: Jannet
"""
# Load libraries
# Load libraries
import os
import pandas as pd
from pandas import read_csv

######################
#### Data Import #####
######################
# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

# import packages from file
import behavior_model_func as bmf

# import datafiles
train =  read_csv('kian_trainset_focal.csv', 
                    header = 0, 
                    index_col = 0)

test =  read_csv('kian_testset_focal.csv', 
                    header = 0, 
                    index_col = 0)

# subset and reorder predictors
train = train[['ID', 'TID', 'track_position', 'track_length', 'focal', 'year', # identifiers
                   'feed', 'rest', 'social', 'travel', # behaviors         
                   'since_rest', 'since_feed', 'since_travel', 'sex', # internal features  
                   'gestation', 'lactation', 'mating', 'nonreproductive', # internal or external features - reproductive state can be both
                   'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count', # external features
                   'years', 'minutes_sin', 'minutes_cos', 'doy_sin','doy_cos', # external/time features
                   'adults', 'infants', 'juveniles', # external/group features
                   'individual_continuity', 'length', 'position']] # sampling features 

test = test[['ID', 'TID', 'track_position', 'track_length', 'focal', 'year', # identifiers
                   'feed', 'rest', 'social', 'travel', # behaviors         
                   'since_rest', 'since_feed', 'since_travel', 'sex', # internal features  
                   'gestation', 'lactation', 'mating', 'nonreproductive', # internal or external features - reproductive state can be both
                   'fragment', 'rain', 'temperature', 'flower_count', 'fruit_count', # external features
                   'years', 'minutes_sin', 'minutes_cos', 'doy_sin','doy_cos', # external/time features
                   'adults', 'infants', 'juveniles', # external/group features
                   'individual_continuity', 'length', 'position']] # sampling features 

# parameters
vrnn_params = {'atype': 'VRNN',
              'mtype': 'GRU',
              'lookback': 22, 
              'n_outputs': 1,
              'predictor':'behavior',
              'hidden_layers': 1,
              'neurons_n': 20,
              'hidden_n0': 50,
              'hidden_n1': 10,
              'learning_rate': 0.001,
              'dropout_rate': 0.1,               
              'loss': True,
              'max_epochs': 100,
              'batch_size': 512,
              'weights_0': 2.5,
              'weights_1': 1,
              'weights_2': 9,
              'weights_3': 5}

# format training and testing data
train_X, train_y, train_dft, test_X, test_y, test_dft = bmf.train_test_format(train, test, vrnn_params)
    
# generate predictions from activity distribution probabilities
act_dist = bmf.null_mod(train_y, test_y, test_dft, 'act_dist22')

# print results 
act_dist.iloc[0,:]

# ID                  88359
# feature        act_dist22
# lookback              -99
# accuracy         0.712085
# precision        0.256086
# recall           0.255005
# f1               0.255182
# accuracy_f       0.837204
# accuracy_r        0.72891
# accuracy_s       0.982109
# accuracy_t       0.875948
# f1_f              0.09127
# f1_r             0.837638
# f1_s             0.025806
# f1_t             0.066012
# precision_f      0.095833
# precision_r      0.825686
# precision_s      0.029412
# precision_t      0.073413
# recall_f         0.087121
# recall_r         0.849942
# recall_s         0.022989
# recall_t         0.059968
# roc_weight            0.5
# roc_micro             0.5
# roc_macro             0.5
# pr_weight            0.25
# pr_macro         0.691172
# cat_loss         0.624536
# accuracy_3       0.814021
# precision_3      0.331644
# recall_3         0.332344
# f1_3              0.33164
# FF                     69
# FR                    658
# FS                     10
# FT                     55
# RF                    588
# RR                   5902
# RS                     48
# RT                    406
# SF                      8
# SR                     71
# SS                      2
# ST                      6
# TF                     55
# TR                    517
# TS                      8
# TT                     37
# F_pred                720
# R_pred               7148
# S_pred                 68
# T_pred              504.0
# KSD_F            0.039782
# KSP_F            0.262443
# KSD_R            0.069423
# KSP_R            0.004136
# KSD_S            0.014821
# KSP_S            0.998965
# KSD_T            0.044462
# KSP_T            0.158577
# F_dprop          0.080742
# R_dprop          0.852704
# S_dprop          0.006873
# T_dprop          0.059681
# Name: 64, dtype: object

# iterate multiple times
for i in range(999):
    null_pred = bmf.null_mod(train_y, test_y, test_dft, 'actdist_22')
    act_dist = pd.concat([act_dist, null_pred], axis = 0, ignore_index =True)
    print (i)

# save results
act_dist.to_csv(path + 'actdist_22_performance_results.csv')

# generate predictions from transition activity distributions (Markov model)
train_label = bmf.to_label(train_y, prob = False) # generate training labels

tmat = bmf.transition_matrix(train_X, train_y) # generate transition matrix

# {0: [0.0929944203347799,
#   0.8350898946063237,
#   0.006819590824550527,
#   0.06509609423434594],
#  1: [0.08635767380848274,
#   0.8535198950590293,
#   0.009510275470048098,
#   0.05061215566243988],
#  2: [0.06329113924050633,
#   0.8734177215189873,
#   0.02531645569620253,
#   0.0379746835443038],
#  3: [0.08691499522445081,
#   0.8299904489016237,
#   0.007640878701050621,
#   0.07545367717287488]}

markov_pred = bmf.markov_null(train_X, train_y, test_X, test_y, test_dft, 'markov_22')

# view results
markov_pred.iloc[0,:]
# feature        markov_22
# lookback             -99
# accuracy         0.71564
# precision         0.2563
# recall          0.255183
# f1              0.255262
# accuracy_f      0.838033
# accuracy_r      0.731517
# accuracy_s      0.980806
# accuracy_t      0.880924
# f1_f            0.090486
# f1_r            0.839541
# f1_s            0.012195
# f1_t            0.078827
# precision_f      0.09564
# precision_r     0.825857
# precision_s     0.012987
# precision_t     0.090717
# recall_f        0.085859
# recall_r        0.853687
# recall_s        0.011494
# recall_t        0.069692
# roc_weight      0.511898
# roc_micro       0.506491
# roc_macro        0.51054
# pr_weight       0.251755
# pr_macro        0.693049
# cat_loss        0.624796
# accuracy_3      0.816825
# precision_3     0.337405
# recall_3        0.336412
# f1_3            0.336285
# FF                    68
# FR                   673
# FS                     8
# FT                    43
# RF                   572
# RR                  5928
# RS                    61
# RT                   383
# SF                     8
# SR                    73
# SS                     1
# ST                     5
# TF                    63
# TR                   504
# TS                     7
# TT                    43
# F_pred               711
# R_pred              7178
# S_pred                77
# T_pred             474.0
# KSD_F           0.048362
# KSP_F           0.099717
# KSD_R           0.072543
# KSP_R           0.002342
# KSD_S            0.00546
# KSP_S                1.0
# KSD_T           0.049142
# KSP_T           0.090453
# F_dprop          0.08137
# R_dprop          0.85193
# S_dprop         0.010009
# T_dprop         0.056691
# Name: 64, dtype: object

# iterate multiple times
for i in range(999):
    null_pred = bmf.markov_null(train_X, train_y, test_X, test_y, test_dft, 'markov_22')
    markov_pred = pd.concat([markov_pred, null_pred], axis = 0, ignore_index =True)
    print (i)

# save results
markov_pred.to_csv(path + 'markov_22_performance_results.csv')
