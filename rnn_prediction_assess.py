# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:24:30 2024

@author: Jannet

Assess RNN model performance. This code is implemented within the testing phase of the project.

"""
# Load libraries
import os
from pandas import read_csv
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

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

# load parameters
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
# accuracy       0.618483
# precision      0.319312
# recall         0.344875
# f1                0.315
# accuracy_f     0.841232
# accuracy_r     0.668365
# accuracy_s     0.928199
# accuracy_t     0.799171
# f1_f           0.283422
# f1_r           0.772198
# f1_s           0.006557
# f1_t           0.197823
# precision_f    0.245826
# precision_r    0.887891
# precision_s    0.003824
# precision_t    0.139706
# recall_f       0.334596
# recall_r        0.68318
# recall_s       0.022989
# recall_t       0.338736
# roc_weight     0.705982
# roc_micro      0.740802
# roc_macro      0.742573
# pr_weight      0.394006
# pr_macro       0.825516
# cat_loss       0.629123
# accuracy_3     0.769589
# precision_3    0.424474
# recall_3       0.452171
# f1_3           0.417815
# FF                  265
# FR                  277
# FS                   49
# FT                  201
# RF                  687
# RR                 4744
# RS                  445
# RT                 1068
# SF                   14
# SR                   53
# SS                    2
# ST                   18
# TF                  112
# TR                  269
# TS                   27
# TT                  209
# F_pred             1078
# R_pred             5343
# S_pred              523
# T_pred             1496
# KSD_F          0.141186
# KSP_F               0.0
# KSD_R          0.329953
# KSP_R               0.0
# KSD_S          0.209048
# KSP_S               0.0
# KSD_T           0.25741
# KSP_T               0.0
# F_prop         0.137668
# R_prop         0.631657
# S_prop         0.060561
# T_prop         0.170114
# Name: 0, dtype: object

# run permutation sensitivity analysis
perm_df = bmf.perm_behavior(results['model'], 
                     perm_df, 
                     results['test_X'], 
                     results['test_dft'], 
                     results['test_y'], 
                     results['y_label'], 
                     seed = seed,
                     name = 'perm_imp_vrnn_catloss', 
                     path = path,
                     prob = False)

# print sample of output
perm_df.head()
#     feature lookback  accuracy  ...    R_prop    S_prop    T_prop
# 0  original      NaN  0.618483  ...  0.631657  0.060561  0.170114
# 1  behavior        0  0.622749  ...  0.805575       0.0  0.102174
# 2  behavior        1  0.614218  ...  0.805575       0.0  0.102206
# 3  behavior        2  0.616706  ...  0.805575       0.0  0.102174
# 4  behavior        3  0.611493  ...  0.805575       0.0  0.102174

# visualize results



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