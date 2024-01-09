# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:24:30 2024

@author: Jannet

Assess RNN model performance. This code is implemented within the testing phase of the project.

"""
# Load libraries
import os, random
from pandas import read_csv

######################
#### Data Import #####
######################
# set working directory
os.chdir("C:\\behavioral_forecasting_RNN")
# set directory for saving results
path = "C:\\behavioral_forecasting_RNN\\outputs\\"

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

# assign parameters
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

# seet random seed 
seed = 10
random.seed(seed)

results = bmf.model_assess(train = train,
                           test = test,
                           params = vrnn_params)

# each a sense of output 
results.keys()
# dict_keys(['model', 'history', 'confusion_matrix', 
#            'report', 'predictions', 'train_X', 
#            'train_y', 'test_X', 'test_y', 
#            'y_pred', 'y_prob', 'evals', 'params'])

# examine performance metrics
results['report']
#               precision    recall  f1-score      support
# 0              0.246216  0.308081  0.273696   792.000000
# 1              0.888285  0.682460  0.771887  6944.000000
# 2              0.013699  0.080460  0.023411    87.000000
# 3              0.129133  0.335494  0.186486   617.000000
# accuracy       0.615758  0.615758  0.615758     0.615758
# macro avg      0.319333  0.351624  0.313870  8440.000000
# weighted avg   0.763521  0.615758  0.674627  8440.000000

# get daily proportion of behaviors
daily_prob = bmf.daily_dist(results['predictions'])

# plot results 
daily_plot = bmf.daily_dist_plot(daily_prob)
daily_plot.savefig(path+'rnn_val_f1max_daily_prob_plot.jpg', dpi=150) # save monitoring plot and examine in output file

# permutation feature importance sensitivity analysis
# permuate each feature in each lookback to see how the feature x lookback affects the prediction outcomes
perm_df = bmf.algo_var(results['test_y'], 
                  results['test_dft'], 
                  results['y_label'], 
                  results['y_pred'], 
                  results['y_predmax'], 
                  results['y_prob'], 
                  'original', 
                  prob = True)

# View original metrics 
perm_df.iloc[0,:]
# feature        original
# lookback            NaN
# accuracy       0.619787
# precision      0.323629
# recall         0.356786
# f1             0.320388
# accuracy_f     0.840166
# accuracy_r     0.671209
# accuracy_s     0.926659
# accuracy_t      0.80154
# f1_f           0.288877
# f1_r           0.773377
# f1_s           0.012759
# f1_t           0.206537
# precision_f    0.247964
# precision_r    0.893228
# precision_s    0.007407
# precision_t    0.145917
# recall_f        0.34596
# recall_r       0.681884
# recall_s       0.045977
# recall_t       0.353323
# roc_weight     0.710482
# roc_micro      0.744513
# roc_macro      0.744888
# pr_weight      0.393019
# pr_macro       0.825158
# cat_loss       0.629754
# accuracy_3     0.770972
# precision_3    0.429036
# recall_3       0.460389
# f1_3            0.42293
# FF                  274
# FR                  274
# FS                   38
# FT                  206
# RF                  702
# RR                 4735
# RS                  455
# RT                 1052
# SF                   10
# SR                   55
# SS                    4
# ST                   18
# TF                  119
# TR                  237
# TS                   43
# TT                  218
# F_pred             1105
# R_pred             5301
# S_pred              540
# T_pred             1494
# KSD_F          0.141186
# KSP_F               0.0
# KSD_R          0.354914
# KSP_R               0.0
# KSD_S          0.226989
# KSP_S               0.0
# KSD_T           0.26209
# KSP_T               0.0
# F_dprop         0.137101
# R_dprop         0.615099
# S_dprop         0.069351
# T_dprop         0.178449
# Name: 0, dtype: object

filename = vrnn_params['atype'] + '_' + vrnn_params['mtype'] + '_' + vrnn_params['predictor'] + '_catloss_' + str(seed)
# run permutation sensitivity analysis
perm_df = bmf.perm_importance(results['model'], 
                              vrnn_params,
                              perm_df, 
                              results['test_X'], 
                              results['test_dft'], 
                              results['test_y'], 
                              results['y_label'], 
                              seed = seed,
                              name = filename, 
                              path = path,
                              prob = False)

# print sample of output
perm_df.head()
#    ID   feature lookback  accuracy  ...   F_dprop   R_dprop   S_dprop   T_dprop
# 0  10  original      -99  0.619787  ...  0.137101  0.615099  0.069351  0.178449
# 1  10  behavior        0  0.616943  ...  0.094219  0.801049  0.011076  0.093657
# 2  10  behavior        1  0.624289  ...  0.094609  0.800024  0.010511  0.094856
# 3  10  behavior        2  0.618957  ...  0.094969  0.800967  0.010887  0.093177
# 4  10  behavior        3  0.613152  ...  0.094581  0.800629  0.010748  0.094041

# use wrapper to evaluate model and conduct permutation feature importance
rnn_perm = bmf.eval_pipeline(train, 
                             test, 
                             vrnn_params, 
                             path)

# visualize results
permimp_plot = bmf.pi_plot(rnn_perm, ['accuracy','f1','precision','recall'])
permimp_plot.savefig(path+filename + '_perm_importance_plot.jpg', dpi=150) # save monitoring plot and examine in output file